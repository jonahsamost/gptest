import random
import json
import time
import csv
import yaml
import shutil
import zipfile
import tempfile
import os
from gptest.utils.utils import get_base_dir, download_file_with_lock
from gptest.utils.ddp_utils import log0, print0
from gptest.evals.core_eval import evaluate_task

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, 'eval_bundle')
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, 'eval_bundle')
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Place eval_bundle directory at {eval_bundle_dir}")



def evaluate_model(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.
    - max_per_task: crop the data to this many examples per task for testing (-1 = disable)
    """
    # Load config and task metadata
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle to disk (and unzip if needed)
    if not os.path.exists(eval_bundle_dir):
        log0("Downloading eval bundle")
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
        log0("Finished downloading eval bundle")
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    if not os.path.exists(config_path):
        place_eval_bundle(os.path.join(EVAL_BUNDLE_URL, "eval_bundle.zip"))
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values from eval metadata
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        log0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})...")

        # Load data for this task
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # shuffle the data because in many cases it appears ordered but we want
        # the ability to only run a subset of the data for debugging purposes etc.
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # run the evaluation for this task
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        log0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out