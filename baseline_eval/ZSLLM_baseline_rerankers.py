import os
import time
import requests
import pandas as pd
import argparse
import json
import re
import string
from dataclasses import dataclass
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
# vLLM for the final answer generation
from vllm import LLM, SamplingParams
import subprocess
from contextlib import contextmanager

# --- Configuration ---
LOCAL_PORT = 8888
REMOTE_PORT = 8000
SSH_SERVER = "kdh0901@bass.snu.ac.kr"


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """
    Extract the string inside the <answer> and </answer> tags,
    as well as the strings before and after the tags.
    """

    answer_pattern = r"<answer>(.*?)</answer>"
    match_iter = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match_iter)

    # If there are 0 or exactly 1 matches, return None for all parts
    if len(matches) == 0:
        return None
    # If there are 2 or more matches, use the last one
    last_match = matches[-1]
    inside = last_match.group(1).strip()

    return inside


@contextmanager
def ssh_tunnel(local_port, remote_port, ssh_server):
    """A context manager to establish and clean up an SSH tunnel."""

    # The SSH command to start the tunnel
    ssh_command = [
        "ssh",
        "-L",
        f"{local_port}:localhost:{remote_port}",
        ssh_server,
        "-N",  # Do not execute a remote command
    ]

    tunnel_process = None
    try:
        # 1. SETUP: Start the SSH tunnel process
        print("[*] Starting SSH tunnel...")
        tunnel_process = subprocess.Popen(ssh_command)

        # Give the tunnel a moment to establish
        time.sleep(2)

        # Check if the process started correctly
        if tunnel_process.poll() is not None:
            raise RuntimeError(
                "SSH tunnel failed to start. Check credentials/connection."
            )

        print(
            f"[*] Tunnel active: localhost:{local_port} -> {ssh_server}:{remote_port}"
        )

        # 2. YIELD: The code inside the 'with' block runs here
        yield

    finally:
        # 3. TEARDOWN: This code runs after the 'with' block finishes or fails
        if tunnel_process:
            print("\n[*] Terminating the SSH tunnel process.")
            tunnel_process.terminate()
            tunnel_process.wait()
            print("[*] Tunnel closed.")


# --- 1. Configuration ---
# All settings are in one place for easy modification.
CONFIG = {
    "data": {
        "path": "./data/nq/test.parquet",
        "question_column": "question",
        "answer_column": "golden_answers",
    },
    "retriever": {
        "url": "http://localhost:8888/retrieve",  # Assumes your retriever is running on port 8000
        "top_k_initial": 50,
    },
    "reranker_server": {
        "url": "http://localhost:8001/rerank"  # The URL for our new reranker server
    },
    "rerankers_to_test": {
        # 'id': (is_reasoning_model?)
        "baseline": False,
        "rank_t5": False,
        "rank_zephyr": False,
        "rank_r1": False,
        "rearank": False,
        "rank1": True,
        "reasonrank": False,
    },
    "generator": {
        "model_path": "Qwen/Qwen2.5-3B-Instruct",
        "tensor_parallel_size": 2,
    },
    "experiment": {
        "top_k_final": 3,
        "output_file_prefix": "results",
        "limit_questions": None,
        "batch_size": 512,
    },
}

# --- 2. Data Structures ---
# Using dataclasses to keep the data organized.


@dataclass
class Document:
    id: str
    text: str
    score: float


@dataclass
class RerankResult:
    documents: List[Document]
    reasonings: List[str] = None


# --- 3. Component Implementations ---


class APIRetriever:
    """Handles communication with the remote retriever server."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def batch_retrieve(self, queries: List[str], top_k: int) -> List[Document]:
        """Retrieves documents for a single query."""
        payload = {"queries": queries, "topk": top_k, "return_scores": True}
        retries = 3
        for i in range(retries):
            try:
                response = requests.post(self.config["url"], json=payload)
                response.raise_for_status()
                # Assuming the response format is like: {"result": [[{"document": {"contents": "...", "id": ...}, "score": ...}]]}
                results = response.json()["result"]
                batched_documents = []
                for res in results:
                    documents = []
                    for j, r in enumerate(res):
                        documents.append(
                            Document(
                                id=r["document"].get("id", str(j)),
                                text=r["document"]["contents"],
                                score=r["score"],
                            )
                        )
                    batched_documents.append(documents)
                return batched_documents
            except requests.exceptions.RequestException as e:
                print(f"Retriever request failed: {e}. Retrying ({i + 1}/{retries})...")
                if i < retries - 1:
                    time.sleep(0.5)
                else:
                    print(f"Failed to retrieve for query: {queries}")
                    return []  # Return empty list on failure


class VLLMGenerator:
    """A wrapper for the vLLM generation engine."""

    def __init__(self, config: Dict[str, Any]):
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
        self.llm = LLM(
            model=config["model_path"],
            tensor_parallel_size=config["tensor_parallel_size"],
        )
        self.sampling_params = SamplingParams(
            temperature=1, top_p=0.95, max_tokens=2048
        )

    def generate(self, prompts: List[str]) -> List[str]:
        """Generates answers for a batch of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]


# --- 4. Reranker Abstraction and Implementations ---


class BaseReranker(ABC):  # Base class is still useful
    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.config = config

    @abstractmethod
    def rerank(self, queries: List[str], documents: List[Document]) -> RerankResult:
        pass

    @property
    def has_reasoning(self) -> bool:
        return False


class APIReranker(BaseReranker):
    """A client-side reranker that calls the reranker server."""

    def __init__(self, model_id: str, config: Dict[str, Any], is_reasoning_model: bool):
        super().__init__(model_id, config)
        self._has_reasoning = is_reasoning_model

    def rerank(
        self, queries: List[str], documents_batch: List[List[Document]]
    ) -> List[RerankResult]:
        print(
            f"Calling reranker server for model '{self.model_id}' with a batch of {len(queries)} queries..."
        )
        documents_for_payload = [
            [doc.__dict__ for doc in doc_list] for doc_list in documents_batch
        ]
        payload = {
            "queries": queries,
            "documents_batch": documents_for_payload,
            "model_name": self.model_id,
        }
        retries = 3
        for i in range(retries):
            try:
                response = requests.post(
                    self.config["reranker_server"]["url"], json=payload
                )
                response.raise_for_status()
                data = response.json()  # Expects a list of results

                results = []
                for item in data:
                    reranked_docs = [Document(**d) for d in item.get("documents", [])]
                    reasonings = item.get("reasonings", [])
                    results.append(
                        RerankResult(
                            documents=reranked_docs[
                                : self.config["experiment"]["top_k_final"]
                            ],
                            reasonings=reasonings[: self.config["experiment"]["top_k_final"]] if self._has_reasoning else None,
                        )
                    )
                return results
            except requests.exceptions.RequestException as e:
                print(f"Retriever request failed: {e}. Retrying ({i + 1}/{retries})...")
                if i < retries - 1:
                    time.sleep(0.5)
                else:
                    raise RuntimeError("API call to reranker server failed")

    @property
    def has_reasoning(self) -> bool:
        return self._has_reasoning


class BaselineReranker(BaseReranker):  # Baseline is still local
    def rerank(
        self, queries: List[str], documents_batch: List[List[Document]]
    ) -> List[RerankResult]:
        print(f"Running Baseline for a batch of {len(queries)} queries...")
        top_k = self.config["experiment"]["top_k_final"]
        return [RerankResult(documents=docs[:top_k]) for docs in documents_batch]


# --- 5. The Main Experiment Runner ---


class RAGExperimentRunner:
    def __init__(self, config: Dict[str, Any], reranker_name: str):
        self.config = config
        self.reranker_name = reranker_name

        # Make output file directory & Modify output file name to be specific to this run
        os.makedirs("./baseline_results", exist_ok=True)
        self.config["experiment"]["output_file"] = (
            f"./baseline_results/results_{reranker_name}.jsonl"
        )

        self.retriever = APIRetriever(config["retriever"])
        self.generator = VLLMGenerator(config["generator"])

        if reranker_name != "baseline":
            # For this to work, you need a way to know if a model has reasoning.
            # A simple dictionary lookup is fine for this.
            reasoning_models = ["rank1"]
            is_reasoning = reranker_name in reasoning_models
            self.reranker = APIReranker(reranker_name, config, is_reasoning)
        else:
            self.reranker = BaselineReranker("baseline", config)

    def _format_prompt(
        self, query: str, documents: List[Document], reasonings: List[str] = None
    ) -> str:
        """Formats the final prompt for the generator LLM."""

        if reasonings:
            context = "" 
            for idx, (doc_item, reasoning) in enumerate(zip(documents, reasonings)):
                content = doc_item.text.strip()
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                context += f"Doc {idx + 1}(Title: {title}) {text}\nReasoning about Doc {idx + 1}'s relevance to the query: {reasoning}\n"
            prompt = f"""You are a meticulous and expert research assistant. Your task is to answer the user's question based on the provided documents and the information about their relevance to the query.

Follow these two steps precisely:
1. First, write a step-by-step reasoning that explains how you will use the documents to derive the final answer.
2. Then, on a new line, provide ONLY the final, concise answer enclosed in <answer> and </answer> tags. Do not add any explanation around the tags. Example: <answer>Beijing</answer>

Question: {query}
Documents and their relevance to the query:
{context}
"""
        else:
            context = ""
            for idx, doc_item in enumerate(documents):
                content = doc_item.text.strip()
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                context += f"Doc {idx + 1}(Title: {title}) {text}\n"
            prompt = f""""You are a meticulous and expert research assistant. Your task is to answer the user's question based on the provided documents and the information about their relevance to the query.

Follow these two steps precisely:
1. First, write a step-by-step reasoning that explains how you will use the documents to derive the final answer.
2. Then, on a new line, provide ONLY the final, concise answer enclosed in <answer> and </answer> tags. Do not add any explanation around the tags. Example: <answer>Beijing</answer>

Question: {query}
Documents:
{context}
"""
        return prompt

    def run(self):
        df = pd.read_parquet(self.config["data"]["path"])
        if self.config["experiment"]["limit_questions"]:
            df = df.head(self.config["experiment"]["limit_questions"])

        batch_size = self.config["experiment"]["batch_size"]
        all_results = []

        # Iterate through the dataframe in batches
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            batch_queries = batch_df[self.config["data"]["question_column"]].tolist()
            batch_golden_answers = batch_df[
                self.config["data"]["answer_column"]
            ].tolist()

            print(
                f"\n\n{'=' * 20} Processing Batch {i // batch_size + 1} (Queries {i + 1}-{i + len(batch_df)}) {'=' * 20}"
            )

            # 1. Batched Retrieval
            batch_retrieved_docs = self.retriever.batch_retrieve(
                batch_queries, self.config["retriever"]["top_k_initial"]
            )

            batch_prompts_to_generate = []
            batch_prompt_metadata = []

            # 2. Batched Reranking (iterate through reranker strategies)
            batch_rerank_results = self.reranker.rerank(
                batch_queries, batch_retrieved_docs
            )

            # Iterate through each query *within the batch* to prepare prompts
            for query_idx, (query, golden_answers) in enumerate(
                zip(batch_queries, batch_golden_answers)
            ):
                rerank_result = batch_rerank_results[query_idx]
                top_k_docs = rerank_result.documents

                # A. Prompt WITHOUT reasoning
                prompt_no_reasoning = self._format_prompt(query, top_k_docs)
                messages = [{"role": "user", "content": prompt_no_reasoning}]
                text = self.generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_prompts_to_generate.append(text)
                context = ""
                for idx, doc_item in enumerate(top_k_docs):
                    content = doc_item.text.strip()
                    title = content.split("\n")[0]
                    text = "\n".join(content.split("\n")[1:])
                    context += f"Doc {idx + 1}(Title: {title}) {text}\n"

                batch_prompt_metadata.append(
                    {
                        "original_query_index": i + query_idx,
                        "query": query,
                        "reranker": self.reranker_name,
                        "variation": "no_reasoning",
                        "context": context,
                        "golden_answers": golden_answers.tolist(),
                    }
                )

                # B. Prompt WITH reasoning
                if self.reranker.has_reasoning and rerank_result.reasonings:
                    prompt_with_reasoning = self._format_prompt(
                        query, top_k_docs, rerank_result.reasonings
                    )
                    messages = [{"role": "user", "content": prompt_with_reasoning}]
                    text = self.generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    batch_prompts_to_generate.append(text)
                    batch_prompt_metadata.append(
                        {
                            "original_query_index": i + query_idx,
                            "query": query,
                            "reranker": self.reranker_name,
                            "variation": "with_reasoning",
                            "context": context,
                            "reasonings": rerank_result.reasonings,
                            "golden_answers": golden_answers.tolist(),
                        }
                    )

            # 3. Batched Generation (one big vLLM call per batch of queries)
            if not batch_prompts_to_generate:
                continue

            print(
                f"Generating {len(batch_prompts_to_generate)} responses for this batch..."
            )
            generated_responses = self.generator.generate(batch_prompts_to_generate)

            # 4. Collect and Save Results for the Batch
            # Create a placeholder for this batch's results, indexed by original query index

            batch_results_list = []
            for response_idx, response in enumerate(generated_responses):
                meta = batch_prompt_metadata[response_idx]
                correctness = em_check(
                    normalize_answer(extract_solution(response) or ""),
                    meta["golden_answers"],
                )

                retrieval_correctness = False
                for golden_answer in meta.get("golden_answers"):
                    if normalize_answer(golden_answer) in normalize_answer(meta["context"]):
                        retrieval_correctness = True
                        break

                batch_results_list.append(
                    {
                        "query": meta["query"],
                        "response": response,
                        "context": meta["context"],
                        "reasonings": meta.get("reasonings", []),
                        "golden_answers": meta.get("golden_answers"),
                        "correctness": correctness,
                        "retrieval_correctness": retrieval_correctness,
                    }
                )

            # Append to master list and save to file
            all_results.extend(batch_results_list)
            if self.reranker.has_reasoning and rerank_result.reasonings:
                reasoning_results_list = [
                    result for result in batch_results_list if result.get("reasonings")
                ]
                non_reasoning_results_list = [
                    result
                    for result in batch_results_list
                    if not result.get("reasonings")
                ]
                with open(
                    self.config["experiment"]["output_file"].replace(
                        ".jsonl", "_with_reasoning.jsonl"
                    ),
                    "a",
                ) as f:
                    for result in reasoning_results_list:
                        f.write(json.dumps(result) + "\n")
                with open(
                    self.config["experiment"]["output_file"].replace(
                        ".jsonl", "_no_reasoning.jsonl"
                    ),
                    "a",
                ) as f:
                    for result in non_reasoning_results_list:
                        f.write(json.dumps(result) + "\n")
            else:
                with open(self.config["experiment"]["output_file"], "a") as f:
                    for result in batch_results_list:
                        f.write(json.dumps(result) + "\n")
            print(f"Saved results for batch {i // batch_size + 1}.")

        print(
            f"\n\nExperiment finished. Results saved to {self.config['experiment']['output_file']}"
        )
        return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a RAG experiment for a single reranker."
    )
    parser.add_argument(
        "--reranker-name",
        type=str,
        required=True,
        help="The name of the reranker to test (e.g., 'baseline', 'rank_t5').",
    )
    args = parser.parse_args()

    # Make sure to include all class definitions (Document, RAGExperimentRunner, etc.) in the file
    runner = RAGExperimentRunner(CONFIG, args.reranker_name)
    with ssh_tunnel(LOCAL_PORT, REMOTE_PORT, SSH_SERVER):
        runner.run()
