import uvicorn
import argparse
import os
import signal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional
from llmrankers.setwise import RankR1SetwiseLlmRanker
from llmrankers.rankers import SearchResult

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
    reasoning: List[str] = None


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
            "ielabgroup/Rank-R1-7B-v0.1",
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


class RankR1Reranker(ReasoningReranker):
    def __init__(self, model_id):
        super().__init__(model_id)
        prompt_file_path = "~/Reranker/baseline_eval/baselines/llm-rankers/Rank-R1/prompts/prompt_setwise-R1.toml"
        extended_path = os.path.expanduser(prompt_file_path)
        self.ranker = RankR1SetwiseLlmRanker(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            lora_name_or_path=model_id,
            prompt_file=extended_path,
            num_child=19,
            k=1,
            verbose=False,
        )

    def rerank(
        self, queries: List[str], documents_batch: List[List[Document]]
    ) -> List[RerankResult]:
        final_results = []
        for query, documents in zip(queries, documents_batch):
            id_to_text = {doc.id: doc.text for doc in documents}
            docs = [
                SearchResult(docid=doc.id, text=doc.text, score=None)
                for doc in documents
            ]
            rerank_result = self.ranker.rerank(query, docs)
            documents = [
                Document(
                    id=search_result.docid,
                    text=id_to_text[search_result.docid],
                    score=search_result.score,
                )
                for search_result in rerank_result
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
        "rank_r1": "ielabgroup/Rank-R1-7B-v0.1",  # Placeholder
    }

    if args.model_name not in RERANKER_PATHS:
        raise ValueError(
            f"Unknown model name: {args.model_name}. Must be one of {list(RERANKER_PATHS.keys())}"
        )

    model_path = RERANKER_PATHS[args.model_name]

    print(f"Loading single reranker model: {args.model_name} from {model_path}")
    reranker_model = RankR1Reranker(model_path)
    print(f"Model {args.model_name} loaded. Starting server...")

    uvicorn.run(app, host="0.0.0.0", port=8001)
