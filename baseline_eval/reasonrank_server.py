import os
import uvicorn
import argparse
import signal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional
from types import SimpleNamespace
from .baselines.ReasonRank.data import Query as ReasonRankQuery
from .baselines.ReasonRank.data import Request as ReasonRankRequest
from .baselines.ReasonRank.data import Candidate as ReasonRankCandidate
from .baselines.ReasonRank.rerank.rank_listwise_os_llm import RankListwiseOSLLM
from .baselines.ReasonRank.rerank.reranker import Reranker

# --- Imports from previous skeleton ---
from dataclasses import dataclass
from abc import ABC, abstractmethod


# --- 2. Pydantic Models for API Contract ---
class DocumentIn(BaseModel):
    id: str
    text: str
    score: float


class RerankRequest(BaseModel):
    queries: List[str]
    documents_batch: List[List[DocumentIn]]
    model_name: str  # e.g., "rank_t5", "rank_r1"


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


class ReasoningReranker(BaseReranker):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        print(f"Initializing SKELETON for reasoning reranker: {model_id}")
        if model_id not in [
            "rank_r1",
            "le723z/Rearank-7B",
            "jhu-clsp/rank1-7B",
            "liuwenhan/reasonrank-7B",
        ]:
            raise ValueError(f"Unknown reasoning model: {model_id}")

    def rerank(
        self, queries: List[str], documents_batch: List[List[Document]]
    ) -> List[RerankResult]:
        raise NotImplementedError(
            "Reasoning rerankers must implement the rerank method."
        )

    @property
    def has_reasoning(self) -> bool:
        return True


class ReasonRankReranker(ReasoningReranker):
    def __init__(self, model_id):
        super().__init__(model_id)
        prompt_info_path = (
            "~/Reranker/baseline_eval/baselines/ReasonRank/listwise_prompt_r1.toml"
        )
        expanded_path = os.path.expanduser(prompt_info_path)
        self.args = SimpleNamespace(
            model_path=model_id,
            window_size=20,
            step_size=10,
            retrieval_num=50,
            num_passes=1,
            reasoning_maxlen=3072,
            use_gpt4cot_retrieval=True,
            shuffle_candidates=False,
            prompt_mode="rank_GPT_reasoning",
            context_size=32768,
            variable_passages=True,
            vllm_batched=True,
            batch_size=512,
            num_gpus=2,
            prompt_info_path=expanded_path,
            notes="",
            lora_path=None,
            rerank_topk=None,
            max_lora_rank=32,
        )
        self.agent = RankListwiseOSLLM(
            args=self.args,
            model=self.args.model_path,
            context_size=self.args.context_size,
            prompt_mode=self.args.prompt_mode,
            num_gpus=self.args.num_gpus,
            window_size=self.args.window_size,
            prompt_info_path=self.args.prompt_info_path,
            vllm_batched=self.args.vllm_batched,
        )
        self.reranker = Reranker(self.agent)

    def rerank(
        self, queries: List[str], documents_batch: List[List[Document]]
    ) -> List[RerankResult]:
        retrieved_results = []
        for i, (query, documents) in enumerate(zip(queries, documents_batch)):
            query = ReasonRankQuery(qid=str(i), text=query)
            candidates = [
                ReasonRankCandidate(
                    docid=str(doc.id), score=doc.score, doc={"text": doc.text}
                )
                for doc in documents
            ]
            retrieved_results.append(
                ReasonRankRequest(query=query, candidates=candidates)
            )
        reranked_batch, time_cost, rerank_details_batch = self.reranker.rerank_batch(
            retrieved_results,
            rank_start=0,
            rank_end=self.args.retrieval_num
            if self.args.rerank_topk is None
            else self.args.rerank_topk,
            window_size=min(
                self.args.window_size, len(retrieved_results[0].candidates)
            ),
            shuffle_candidates=self.args.shuffle_candidates,
            step=self.args.step_size,
            vllm_batched=self.args.vllm_batched,
        )

        final_results = []
        for rerank_result in reranked_batch:
            documents = [
                Document(
                    id=candidate.docid,
                    text=candidate.doc["text"],
                    score=candidate.score,
                )
                for candidate in rerank_result.candidates
            ]
            final_results.append(RerankResult(documents=documents))
        return final_results


# --- 4. FastAPI Application Setup ---
app = FastAPI()
reranker_model: Optional[Any] = None


@app.post("/rerank")
def rerank_endpoint(request: RerankRequest):
    """Endpoint to rerank documents using the loaded model."""
    if reranker_model is None:
        raise HTTPException(status_code=503, detail="Reranker model not loaded.")
    docs_to_process = [
        [Document(**d.dict()) for d in documents]
        for documents in request.documents_batch
    ]
    results = reranker_model.rerank(request.queries, docs_to_process)

    return [
        {
            "documents": [d.__dict__ for d in result.documents],
            "reasonings": result.reasoning,
        }
        for result in results
    ]
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
        "--model-name",
        type=str,
        required=True,
        help="The name of the reranker to load (e.g., 'rank_t5', 'rank_r1').",
    )
    args = parser.parse_args()

    # Configuration for model paths
    RERANKER_PATHS = {
        "reasonrank": "liuwenhan/reasonrank-7B",  # Placeholder
    }

    if args.model_name not in RERANKER_PATHS:
        raise ValueError(
            f"Unknown model name: {args.model_name}. Must be one of {list(RERANKER_PATHS.keys())}"
        )

    model_path = RERANKER_PATHS[args.model_name]

    print(f"Loading single reranker model: {args.model_name} from {model_path}")
    reranker_model = ReasonRankReranker(model_path)

    print(f"Model {args.model_name} loaded. Starting server...")

    uvicorn.run(app, host="0.0.0.0", port=8001)
