import uvicorn
import argparse
import os
import signal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional
from .baselines.rank1.rank1 import rank1

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
            "jhu-clsp/rank1-7b",
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


class Rank1Reranker(ReasoningReranker):
    def __init__(self, model_id):
        super().__init__(model_id)
        self.reranker = rank1(
            model_name_or_path=model_id,
            num_gpus=2,
            device="cuda",
            context_size=16000,
            max_output_tokens=8192,
            fp_options="float16",
        )

    def rerank(
        self, queries: List[str], documents_batch: List[List[Document]]
    ) -> List[RerankResult]:
        retrieved_results = []
        for i, (query, documents) in enumerate(zip(queries, documents_batch)):
            queries = [query] * len(documents)
            corpus = [doc.text for doc in documents]
            reasonings, scores = self.reranker.predict(list(zip(queries, corpus)))
            # Sort documents and reasoning by scores
            sorted_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            sorted_documents = [documents[i] for i in sorted_indices]
            sorted_reasonings = [reasonings[i] for i in sorted_indices]
            retrieved_results.append(
                RerankResult(documents=sorted_documents, reasoning=sorted_reasonings)
            )

        return retrieved_results


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
        "rank1": "jhu-clsp/rank1-7b",  # Placeholder
    }

    if args.model_name not in RERANKER_PATHS:
        raise ValueError(
            f"Unknown model name: {args.model_name}. Must be one of {list(RERANKER_PATHS.keys())}"
        )

    model_path = RERANKER_PATHS[args.model_name]

    print(f"Loading single reranker model: {args.model_name} from {model_path}")
    reranker_model = Rank1Reranker(model_path)

    print(f"Model {args.model_name} loaded. Starting server...")

    uvicorn.run(app, host="0.0.0.0", port=8001)
