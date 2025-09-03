import uvicorn
import argparse
import os
import signal
import re
import asyncio
from tqdm.asyncio import tqdm_asyncio
import openai
import torch
import requests
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional, Dict
from transformers import AutoTokenizer
import subprocess
from contextlib import contextmanager

# --- Imports from previous skeleton ---
from dataclasses import dataclass
from abc import ABC, abstractmethod

LOCAL_PORT = 7999
REMOTE_PORT = 8000
SSH_SERVER = "kdh0901@bass.snu.ac.kr"


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


# --- 2. Pydantic Models for API Contract ---
class DocumentIn(BaseModel):
    id: str
    text: str
    score: float


class RerankRequest(BaseModel):
    query: str


# --- 3. Internal Data Structures & Reranker Logic (same as before) ---
@dataclass
class Document:
    id: str
    text: str
    score: float


@dataclass
class RerankResult:
    documents: List[Document]
    reasoning: Optional[str] = None


class BaseReranker(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> RerankResult:
        pass

    @property
    def has_reasoning(self) -> bool:
        return False


class APIRetriever:
    """Handles communication with the remote retriever server."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        """Retrieves documents for a single query."""
        payload = {"queries": [query], "topk": top_k, "return_scores": True}
        retries = 3
        for i in range(retries):
            try:
                response = requests.post(self.config["url"], json=payload)
                response.raise_for_status()
                # Assuming the response format is like: {"result": [[{"document": {"contents": "...", "id": ...}, "score": ...}]]}
                results = response.json()["result"]
                documents = []
                for j, r in enumerate(results[0]):
                    documents.append(
                        Document(
                            id=r["document"].get("id", str(j)),
                            text=r["document"]["contents"],
                            score=r["score"],
                        )
                    )
                return documents
            except requests.exceptions.RequestException as e:
                print(f"Retriever request failed: {e}. Retrying ({i + 1}/{retries})...")
                if i < retries - 1:
                    time.sleep(0.5)
                else:
                    print(f"Failed to retrieve for query: {query}")
                    return []  # Return empty list on failure


class SplitRAGReranker:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        max_output_tokens: int = 8192,
        max_score: int = 5,
        concurrency: int = 32,
        retriever_url: str = "http://localhost:8888/retrieve",
        retriever_initial_topk: int = 50,
        vllm_url: str = "http://localhost:8001/v1",
        temperature: float = 0.6,
        top_p: float = 0.95,
        **kwargs,
    ):
        self.max_output_tokens = max_output_tokens
        self.model_name_or_path = model_name_or_path
        self.max_score = max_score

        # Initialize tokenizer with max length of
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.score_token_list = [
            self.tokenizer.encode(str(i))[0] for i in range(1, self.max_score + 1)
        ]

        self.retriever_config = {
            "url": retriever_url,
            "top_k_initial": retriever_initial_topk,
        }
        self.retriever = APIRetriever(config=self.retriever_config)

        self.client = openai.AsyncOpenAI(
            api_key="vllm",
            base_url=vllm_url,
        )
        self.semaphore = asyncio.Semaphore(concurrency)

        self.sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_output_tokens,
            "logprobs": 20,
        }

    async def get_completion(
        self, client, model_name, prompt, sampling_params, semaphore
    ):
        async with self.semaphore:
            try:
                response = await client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    **sampling_params,
                )
                return response.choices[0]
            except Exception as e:
                print(f"An error occurred while processing a request: {e}")
                return None

    async def _process_with_vllm(self, prompts):
        """
        vLLM is significantly faster than HF, so we use it by default. This function handles the cases where the model does not generate the end </think> token.

        Args:
            prompts: The prompts to generate from

        Returns:
            outputs: The outputs from the vLLM model
        """
        prompts_with_chat_template = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        tasks = [
            self.get_completion(
                self.client,
                self.model_name_or_path,
                prompt_with_chat_template,
                self.sampling_params,
                self.semaphore,
            )
            for prompt_with_chat_template in prompts_with_chat_template
        ]
        outputs = await tqdm_asyncio.gather(*tasks)

        # Pre-allocate lists with None values
        total_length = len(prompts)
        all_outputs = [None] * total_length
        all_output_token_counts = [None] * total_length
        all_scores = [None] * total_length
        all_reasonings = [None] * total_length

        # Process complete responses first
        for i, output in enumerate(outputs):
            text = output.text
            tokens = output.logprobs.tokens
            token_count = len(tokens)

            pattern = r"<answer>(.*?)\s*</answer>:"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
            else:
                reasoning = None
            all_reasonings[i] = reasoning

            score_token_idx = None
            for idx in range(len(tokens) - 1, len(tokens) - 6, -1):
                if tokens[idx] in set([str(i) for i in range(1, self.max_score + 1)]):
                    score_token_idx = idx
                    break

            if score_token_idx is None:
                all_outputs[i] = text
                all_output_token_counts[i] = token_count
                all_scores[i] = (0, 0)
                continue

            score_integer = int(tokens[score_token_idx])
            logprob_score = output.logprobs.token_logprobs[score_token_idx]

            all_outputs[i] = text
            all_output_token_counts[i] = token_count
            all_scores[i] = (score_integer, logprob_score)

        return all_outputs, all_reasonings, all_output_token_counts, all_scores

    def return_prompt(self, query, doc_content) -> str:
        title = doc_content.split("\n")[0]
        text = "\n".join(doc_content.split("\n")[1:])
        return (
            "Determine if the following passage is relevant to the query. "
            f"First, think step by step and provide the reason why the passage is relevant to the query with the relevance score in 1 to {self.max_score}(The bigger, the more relevant). "
            "Write your final reason of relevance and score in this format after your step by step thought like this format: <answer> {Final reason of relevance} Score: {relevance score} </answer>\n"
            f"Query: {query}\n"
            f"Passage: (Title: {title}) {text}\n"
        )  # force the model to start with this

    def _prepare_prompts_for_rethink(
        self, prompts: List[str], texts: List[str], rethink_text: str = "Wait"
    ) -> List[str]:
        """Prepare prompts for the rethinking step."""
        full_texts = [p + t for p, t in zip(prompts, texts)]
        stripped_texts = [t.split("</think>")[0] for t in full_texts]
        just_generated_texts = [t.split("</think>")[0] for t in full_texts]
        return [s + f"\n{rethink_text}" for s in stripped_texts], just_generated_texts

    @torch.inference_mode()
    async def predict(self, queries, passages, **kwargs):
        """This is setup to run with mteb but can be adapted to your purpose
        input_to_rerank: {"queries": queries, "documents": documents}
        """
        prompts = [
            self.return_prompt(query, passage)
            for query, passage in zip(queries, passages)
        ]
        print(f"Example prompt: ```\n{prompts[0]}\n```")

        texts, reasonings, token_counts, scores = await self._process_with_vllm(prompts)
        return texts, reasonings, scores

    async def rerank(self, query: str) -> RerankResult:
        documents = self.retriever.retrieve(
            query, self.retriever_config["top_k_initial"]
        )
        queries = [query] * len(documents)
        corpus = [doc.text for doc in documents]
        _, reasonings, scores = await self.predict(queries=queries, passages=corpus)
        # Sort documents and reasoning by scores
        for document, score in zip(documents, scores):
            document.score = score
        sorted_indices = sorted(
            range(len(scores)), key=lambda i: (scores[i][0], scores[i][1]), reverse=True
        )
        sorted_documents = [documents[i] for i in sorted_indices]
        sorted_reasonings = [reasonings[i] for i in sorted_indices]

        return RerankResult(documents=sorted_documents, reasoning=sorted_reasonings)


# --- 4. FastAPI Application Setup ---
app = FastAPI()
reranker_model: Optional[Any] = None


@app.post("/rerank")
async def rerank_endpoint(request: RerankRequest):
    """Endpoint to rerank documents using the loaded model."""
    if reranker_model is None:
        raise HTTPException(status_code=503, detail="Reranker model not loaded.")
    result = await reranker_model.rerank(request.query)

    return {
        "documents": [d.__dict__ for d in result.documents],
        "reasonings": result.reasoning,
    }


@app.post("/shutdown")
def shutdown_endpoint():
    """Endpoint to gracefully shut down the Uvicorn server."""
    print("Shutdown request received. Terminating server.")

    # Get the Process ID (PID) of the current process
    pid = os.getpid()

    # Send the SIGINT signal (same as Ctrl+C) to the process
    # Uvicorn is designed to catch this and shut down cleanly.
    os.kill(pid, signal.SIGINT)

    return {"message": "Server is shutting down."}


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch a server for a single reranker model."
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="The name or path of the reranker to load.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="The maximum output token length of the model.",
    )
    parser.add_argument(
        "--max-score",
        type=int,
        default=5,
        help="Maximum scale of relevance score.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="The concurrency for asyncio.",
    )
    parser.add_argument(
        "--retriever-url",
        type=str,
        required=True,
        help="The url for the retriever.",
    )
    parser.add_argument(
        "--retriever-initial-topk",
        type=int,
        default=50,
        help="The number of documents retrieved and ranked by the reranker.",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        required=True,
        help="The url for the vLLM server.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The temperature to use for sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="The top-p sampling probability.",
    )
    args = parser.parse_args()

    print(f"Using single reranker model: {args.model_name_or_path}")
    reranker_model = SplitRAGReranker(
        model_name_or_path=args.model_name_or_path,
        max_output_tokens=args.max_output_tokens,
        max_score=args.max_score,
        concurrency=args.concurrency,
        retriever_url=args.retriever_url,
        retriever_initial_topk=args.retriever_initial_topk,
        vllm_url=args.vllm_url,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"Model {args.model_name_or_path} loaded. Starting server...")

    with ssh_tunnel(LOCAL_PORT, REMOTE_PORT, SSH_SERVER):
        uvicorn.run(app, host="0.0.0.0", port=8002)
