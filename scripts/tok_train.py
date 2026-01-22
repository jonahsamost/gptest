import os
import time
import argparse
import torch
from src.tokenizer import HuggingFaceTokenizer
from src.utils import get_base_dir
from src.dataset import parquets_iter_batched


test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""


def text_iterator(args):
    """
    1 - flatten batches into single iterator
    2 - crop every doc to args.doc_cap chars
    3 - break when we've seen max chars
    """
    nchars = 0
    for i, batch in enumerate(parquets_iter_batched(split='train')):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                print('Hit max chars!')
                return


def create_tok_metrics(tokenizer, tokenizer_dir):
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes = []
    for tid in range(vocab_size):
        tstr = tokenizer.decode([tid])
        if tstr in special_set:
            token_bytes.append(0)
        else:
            id_bytes = len(tstr.encode('utf-8'))
            token_bytes.append(id_bytes)

    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
    token_bytes_path = os.path.join(tokenizer_dir, 'token_bytes.pt')
    with open(token_bytes_path, 'wb') as f:
        torch.save(token_bytes, f)

    tbytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
    print(f'Num special tokens: {len(special_set)}')
    print(f'Token bytes min: {int(tbytes_nonzero.min().item())}')
    print(f'Token bytes max: {int(tbytes_nonzero.max().item())}')
    print(f'Token bytes mean: {tbytes_nonzero.mean().item()}')
    print(f'Token bytes std: {tbytes_nonzero.std().item()}')


def run(args):
    text_iter = text_iterator(args)
    print('Running tokenizer...')
    t0 = time.time()
    tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iter, args.vocab_size)
    train_time = time.time() - t0
    print(f'Training time of tokenizer: {train_time:.2f}s')

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, 'tokenizer')
    tokenizer.save(tokenizer_dir)

    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert test_text == decoded

    create_tok_metrics(tokenizer, tokenizer_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
    parser.add_argument('--max-chars', type=int, default=10_000_000_000, help='Max chars to train on (default: 10B)')
    parser.add_argument('--doc-cap', type=int, default=10_000, help='Max chars per document (default: 10K)')
    parser.add_argument('--vocab-size', type=int, default=32768, help='Vocab size: (default: 2^15)')
    args = parser.parse_args()
    print(f'max chars: {args.max_chars}')
    print(f'doc cap: {args.doc_cap}')
    print(f'vocab size: {args.vocab_size}')
    run(args)