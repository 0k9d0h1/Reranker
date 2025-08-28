THE_INDEX = {
    "dl19": "msmarco-v1-passage",
    "dl20": "msmarco-v1-passage",
    "dl21": "msmarco-v2-passage",
    "dl22": "msmarco-v2-passage",
    "dl23": "msmarco-v2-passage",
    "covid": "beir-v1.0.0-trec-covid.flat",
    "arguana": "beir-v1.0.0-arguana.flat",
    "touche": "beir-v1.0.0-webis-touche2020.flat",
    "news": "beir-v1.0.0-trec-news.flat",
    "scifact": "beir-v1.0.0-scifact.flat",
    "fiqa": "beir-v1.0.0-fiqa.flat",
    "scidocs": "beir-v1.0.0-scidocs.flat",
    "nfc": "beir-v1.0.0-nfcorpus.flat",
    "quora": "beir-v1.0.0-quora.flat",
    "dbpedia": "beir-v1.0.0-dbpedia-entity.flat",
    "fever": "beir-v1.0.0-fever-flat",
    "robust04": "beir-v1.0.0-robust04.flat",
    "signal": "beir-v1.0.0-signal1m.flat",
    "mrtydi-ar": "mrtydi-v1.1-arabic",
    "mrtydi-bn": "mrtydi-v1.1-bengali",
    "mrtydi-fi": "mrtydi-v1.1-finnish",
    "mrtydi-id": "mrtydi-v1.1-indonesian",
    "mrtydi-ja": "mrtydi-v1.1-japanese",
    "mrtydi-ko": "mrtydi-v1.1-korean",
    "mrtydi-ru": "mrtydi-v1.1-russian",
    "mrtydi-sw": "mrtydi-v1.1-swahili",
    "mrtydi-te": "mrtydi-v1.1-telugu",
    "mrtydi-th": "mrtydi-v1.1-thai",
}

THE_TOPICS = {
    "dl19": "dl19-passage",
    "dl20": "dl20-passage",
    "dl21": "dl21-passage",
    "dl22": "dl22-passage",
    "dl23": "dl23-passage",
    "covid": "beir-v1.0.0-trec-covid-test",
    "arguana": "beir-v1.0.0-arguana-test",
    "touche": "beir-v1.0.0-webis-touche2020-test",
    "news": "beir-v1.0.0-trec-news-test",
    "scifact": "beir-v1.0.0-scifact-test",
    "fiqa": "beir-v1.0.0-fiqa-test",
    "scidocs": "beir-v1.0.0-scidocs-test",
    "nfc": "beir-v1.0.0-nfcorpus-test",
    "quora": "beir-v1.0.0-quora-test",
    "dbpedia": "beir-v1.0.0-dbpedia-entity-test",
    "fever": "beir-v1.0.0-fever-test",
    "robust04": "beir-v1.0.0-robust04-test",
    "signal": "beir-v1.0.0-signal1m-test",
    "mrtydi-ar": "mrtydi-v1.1-arabic-test",
    "mrtydi-bn": "mrtydi-v1.1-bengali-test",
    "mrtydi-fi": "mrtydi-v1.1-finnish-test",
    "mrtydi-id": "mrtydi-v1.1-indonesian-test",
    "mrtydi-ja": "mrtydi-v1.1-japanese-test",
    "mrtydi-ko": "mrtydi-v1.1-korean-test",
    "mrtydi-ru": "mrtydi-v1.1-russian-test",
    "mrtydi-sw": "mrtydi-v1.1-swahili-test",
    "mrtydi-te": "mrtydi-v1.1-telugu-test",
    "mrtydi-th": "mrtydi-v1.1-thai-test",
}
DLV2 = ["dl20", "dl21", "dl22", "dl23"]
BRIGHT = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation for DeepRerank")
    parser.add_argument(
        "--model_name",
        type=str,
        default="le723z/v9-s20",
        help="Model name to use for reranking",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for the agent",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for processing (default: 16 * num_gpus)",
    )
    parser.add_argument(
        "--bright", action="store_true", help="Run evaluation on bright datasets"
    )
    parser.add_argument(
        "--standard", action="store_true", help="Run evaluation on standard datasets"
    )
    parser.add_argument(
        "--mrtydi", action="store_true", help="Run evaluation on mrtydi datasets"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip datasets that already have results",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of documents to retrieve with BM25",
    )
    parser.add_argument(
        "--rerank_repeat", type=int, default=0, help="Number of reranking repeats"
    )
    parser.add_argument(
        "--log_name", type=str, default=None, help="Log name for the evaluation"
    )

    return parser.parse_args()


def run_evaluation(args, datasets):
    from rank_gpt import process_rank_results_in_batches, bm25_retrieve
    import json
    from agent import get_agent
    from trec_eval import eval_rerank
    from utils import set_seed, get_results_file
    import os

    set_seed(args.seed)
    agent = get_agent(model_name=args.model_name)

    # Determine batch size based on available GPUs
    num_gpu = (
        len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
        if os.environ.get("CUDA_VISIBLE_DEVICES")
        else 1
    )

    if args.model_name == "deepseek-ai/DeepSeek-R1":
        bs = 1
    else:
        bs = args.batch_size if args.batch_size is not None else 16 * num_gpu

    # Set up results file path``
    model_short_name = args.model_name.split("/")[-1]
    if args.log_name is not None:
        model_short_name = f"{model_short_name}-{args.log_name}"
    if args.rerank_repeat > 0:
        results_file = (
            f"results/{model_short_name}-{args.enable_thinking}-repeat_pass.json"
        )
    else:
        results_file = f"results/{model_short_name}-{args.enable_thinking}.json"
    all_results = get_results_file(results_file)

    for data in datasets:
        # Skip if dataset already has results and skip_existing is True
        if args.skip_existing and data in all_results:
            print(f"Skipping {data} as results already exist")
            continue

        print("#" * 20)
        print(f"Evaluation on {data}")
        print("#" * 20)

        if data in BRIGHT:
            base_dir = os.getcwd()
            bm25_results = get_hits_from_run_bright(base_dir, data)
        else:
            bm25_results = bm25_retrieve(data, top_k_retrieve=100)

        rerank_results = process_rank_results_in_batches(
            agent,
            bm25_results,
            batch_size=bs,
            verbose=False,
            enable_thinking=args.enable_thinking,
        )
        all_metrics, _ = eval_rerank(data, rerank_results)

        if args.rerank_repeat > 0:
            if data not in all_results:
                all_results[data] = {}
            all_results[data]["repeat_0"] = {"metrics": all_metrics}
            for repeat_idx in range(args.rerank_repeat):
                rerank_results = process_rank_results_in_batches(
                    agent,
                    rerank_results,
                    batch_size=bs,
                    verbose=False,
                    enable_thinking=args.enable_thinking,
                )
                all_metrics, _ = eval_rerank(data, rerank_results)

                all_results[data][f"repeat_{repeat_idx + 1}"] = {"metrics": all_metrics}
        else:
            all_results[data] = {"metrics": all_metrics}
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    import os
    from utils import get_hits_from_run_bright

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    args = parse_args()

    if args.mrtydi:
        dataset = [
            "mrtydi-ar",
            "mrtydi-bn",
            "mrtydi-fi",
            "mrtydi-id",
            "mrtydi-ja",
            "mrtydi-ko",
            "mrtydi-ru",
            "mrtydi-sw",
            "mrtydi-te",
            "mrtydi-th",
        ]
    elif args.bright:
        dataset = BRIGHT
    else:
        dataset = [
            "dl19",
            "dl20",
            "covid",
            "nfc",
            "dbpedia",
            "scifact",
            "signal",
            "news",
            "robust04",
        ]

    run_evaluation(args, dataset)

    # Uncomment to run mrtydi evaluation
    # run_mrtydi_evaluation(args)
