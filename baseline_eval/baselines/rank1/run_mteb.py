from __future__ import annotations

import argparse
import mteb
from mteb import MTEB
import logging
import os
import json

from functools import partial
import logging
import math
from typing import Any, Callable, List, Tuple

from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.RetrievalEvaluator import DenseRetrievalExactSearch
from mteb.model_meta import ModelMeta
from mteb.models.rerankers_custom import RerankerWrapper

from prompts import get_prompt, PROMPT_DICT, validate_json, BEIR_DATASETS
from rank1 import rank1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_safe_folder_name(model_name):
    return model_name.replace("/", "_").replace("\\", "_")

def run_evaluation(dataset_name: str, subtask: str, model_name: str, num_gpus: int, skip_prompt: bool) -> None:
    """Run MTEB evaluation for specified model and dataset

    Args:
        dataset_name: Name of MTEB dataset to evaluate
    """
    # Initialize MTEB task and evaluation
    tasks = mteb.get_tasks(tasks=[dataset_name])
    evaluation = MTEB(tasks=tasks)

    if dataset_name == "BrightRetrieval":
        previous_results = f"rank1-run-files/{subtask}_bm25_long_False/score.json"
        eval_splits = ["standard"]
    elif dataset_name == "mFollowIRCrossLingual":
        previous_results = f"rank1-run-files/mFollowIRCrossLingual_{subtask}_predictions.json"
        eval_splits = None
    elif dataset_name == "mFollowIR":
        previous_results = f"rank1-run-files/mFollowIR_{subtask}_predictions.json"
        eval_splits = None
    elif dataset_name in BEIR_DATASETS:
        eval_splits = ["test"]
        previous_results = f"rank1-run-files/{dataset_name}_default_predictions.json"
    else:
        print(f"Running with no subtask or eval splits for dataset: {dataset_name}")
        previous_results = None
        eval_splits = None
    
    encode_kwargs = {
        # use vLLM to batch
        "batch_size": 999999 if "rank1" in model_name.lower() else 32
    }

    prompt = get_prompt(dataset_name, subtask)
    if prompt is not None and not skip_prompt:
        is_prompted = True
    else:
        is_prompted = False
        prompt = None
    print(f"Prompt: {prompt}")
    
    if subtask == "default":
        subtask = None

    if previous_results is not None:
        assert validate_json(previous_results), f"Previous results are not valid json: {previous_results}"
    print(f"Previous results: {previous_results}")

    model = rank1(model_name_or_path=model_name.strip(), num_gpus=num_gpus, dataset_prompt=prompt)
    output_dir = f"results/{model_name}/{dataset_name}_{subtask}"
    print(f"Output directory: {output_dir}")

    # Run evaluation
    evaluation.run(
        model,
        save_predictions=True,
        encode_kwargs=encode_kwargs,
        output_folder=output_dir,
        previous_results=previous_results,
        eval_subsets=[subtask] if previous_results else None,
        eval_splits=eval_splits,
        top_k=100
    )


def main():
    parser = argparse.ArgumentParser(description="Run MTEB evaluation")
    parser.add_argument("-d", "--dataset", required=True, help="MTEB dataset name")
    parser.add_argument("-s", "--subtask", required=False, default=None, help="MTEB subtask name")
    parser.add_argument("-m", "--model_name", required=True, help="Model name")
    parser.add_argument("-n", "--num_gpus", required=False, help="Number of GPUs", default=1)
    parser.add_argument("-p", "--skip_prompt", action="store_true", help="Skip prompt")
    args = parser.parse_args()
    run_evaluation(args.dataset.strip(), args.subtask.strip() if args.subtask is not None else None, args.model_name.strip(), args.num_gpus, args.skip_prompt)


if __name__ == "__main__":
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    main()
