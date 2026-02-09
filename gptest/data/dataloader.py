import torch
import pyarrow.parquet as pq

from gptest.utils.ddp_utils import get_dist_info
from gptest.data.dataset import list_parquet_files
from dataclasses import dataclass, replace

@dataclass
class PQData:
    pq_idx: int = 0
    rg_idx: int = 0
    epoch: int = 0


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    iterator over document batches (list of text strings) from parquet files.

    Handles ddp sharding and approximate resume. Each yield is:
        (text_batch, (pd_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resuming,
    and epoch counts how many times we've cycled through dataset
    """
    ddp = get_dist_info()
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) != 0, 'No dataset parquet files found, did you run dataset.py?'
    parquet_paths = parquet_paths[:-1] if split == 'train' else parquet_paths[-1:]

    if resume_state_dict:
        resume_pq_idx = resume_state_dict['pg_idx'] 
        resume_rg_idx = resume_state_dict['rg_idx']
        resume_epoch = resume_state_dict.get('epoch', 1)
    else:
        resume_pq_idx, resume_rg_idx, resume_epoch = 0, None, 1
    
    first_pass = True
    pg_idx = resume_pq_idx
    epoch = resume_epoch

    while True: # iterate forever
        pg_idx = resume_pq_idx if first_pass else 0
        while pg_idx < len(parquet_paths):
            filepath = parquet_paths[pg_idx]
            pf = pq.ParquetFile(filepath)

            # start from resume point if resuming on same file, else from ddp rank
            if first_pass and (resume_rg_idx is not None) and (pg_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp.world_size
                base_idx += 1  # advance by one to not repeat data after resuming
                rg_idx = base_idx * ddp.world_size + ddp.rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = ddp.rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    pqd = PQData(pq_idx=pq_idx, rg_idx=rg_idx, epoch=epoch)
                    yield batch[i:i+tokenizer_batch_size], pqd
                rg_idx += ddp.world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_dist_data_loader_with_state(
    tokenizer, B, T, split, tokenizer_threads=4, tokenizer_batch_size=128,
    device='cuda', resume_state_dict=None
):
    """
    Stream pretraining data from parquet files, tokenize, yield training batches.

    Streams tokens into flat buffer and reshapes.
    Rows may start mid-document (no guaranteed BOS at position 0)

    Supports appx resume via state_dict
    """
    assert split in ['train', 'val'], "split must in 'train' or 'val'"

    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    needed_tokens = B * T + 1 # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = []

    while True:
        while len(token_buffer) < needed_tokens:
            doc_batch, pqd = next(batches)
            # TODO using HF means this is wrong?
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        tokens = token_buffer[:needed_tokens]
        token_buffer = token_buffer[B*T:]

        use_cuda = device == 'cuda'
        # allocate tensor to page-locked cpu memory (i.e. non-pageable). This enables faster DMA transfers to GPU
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        # non-blocking: on does work if src tensor is pinned in cpu memory and dest is cuda
        # if so:
        #   cpu -> gpu copy is async
        #   python thread does not wait for copy to finish
        #   copy overlaps with gpu compute from previous step
        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)
        yield inputs, targets, pqd


def tokenizing_dist_data_loader(*args, **kwargs):
    for inputs, targets, state_dict in tokenizing_dist_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


def tokenizing_dist_data_loader_with_state_bos(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device='cuda', resume_state_dict=None, buffer_size=1000
):
    """
    BOS-aligned dataloader with best-fit cropping

    reduces token waste compared to simple greedy cropping by searching a buffer
    for docs that fit well, while maintaining 100% utilization (no padding)

    algo per row:
        - from buffered docs, pick largest doc that first entirely
        - repeat until no doc fits
        - when nothing fits, crop a doc to fill remaining space entirely
    
    key properties:
        - every row starts with BOS
        - 100% utilization (no padding, every token gets trained on)
        - appx 35% tokens discarded due to cropping
    """
    assert split in ['train', 'val'], "split must in 'train' or 'val'"
    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pqd = PQData()

    def refill_buffer():
        nonlocal pqd
        doc_batch, pqd = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)
    
    while True:
        rows = []
        for _ in range(B):
            row = []
            while len(row) < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - len(row)

                # find lagest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len
                
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row.extend(doc)
                else:
                    # no doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])

            rows.append(row[:row_capacity])    
        
        use_cuda = device == 'cuda'
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=True)
        inputs = batch_tensor[:, :-1].to(device=device, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, non_blocking=use_cuda)
        yield inputs, targets, replace(pqd)

def tokenizing_dist_data_loader_bos(*args, **kwargs):
    for inputs, targets, state in tokenizing_dist_data_loader_with_state_bos(*args, **kwargs):
        yield inputs, targets
