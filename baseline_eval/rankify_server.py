import uvicorn
import argparse
import os
import signal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional

# --- Imports from previous skeleton ---
from dataclasses import dataclass
from abc import ABC, abstractmethod
from rankify.dataset.dataset import Document as RankifyDocument
from rankify.dataset.dataset import Question as RankifyQuestion
from rankify.dataset.dataset import Context as RankifyContext
from rankify.models.reranking import Reranking


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


class RankifyReranker(BaseReranker):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        print(f"Initializing Rankify reranker: {model_id}")
        if "t5" in self.model_id.lower():
            self.model = Reranking(method="rankt5", model_name=self.model_id)
        elif "zephyr" in self.model_id.lower():
            self.model = Reranking(method="zephyr_reranker", model_name=self.model_id, num_gpus=2)

    def rerank(
        self, queries: List[str], documents_batch: List[List[Document]]
    ) -> List[RerankResult]:
        print(f"Reranking with {self.model_id}...")
        results = []
        for query, documents in zip(queries, documents_batch):
            rankify_question = RankifyQuestion(query)
            rankify_contexts = [
                RankifyContext(text=doc.text, id=int(doc.id)) for doc in documents
            ]
            rankify_document = RankifyDocument(
                question=rankify_question, contexts=rankify_contexts, answers=[]
            )

            self.model.rank([rankify_document])

            documents = []
            for context in rankify_document.reorder_contexts:
                documents.append(
                    Document(id=context.id, text=context.text, score=context.score)
                )

            # id_to_doc_map = {doc.id: doc for doc in documents}
            # reranked_docs = [
            #     id_to_doc_map[ctx.id] for ctx in rankify_document.reorder_contexts
            # ]
            results.append(RerankResult(documents=documents))

        return results


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
        "rank_t5": "rankt5-3b",
        "rank_zephyr": "rank_zephyr_7b_v1_full",
    }

    if args.model_name not in RERANKER_PATHS:
        raise ValueError(
            f"Unknown model name: {args.model_name}. Must be one of {list(RERANKER_PATHS.keys())}"
        )

    model_path = RERANKER_PATHS[args.model_name]

    print(f"Loading single reranker model: {args.model_name} from {model_path}")
    reranker_model = RankifyReranker(model_path)

    print(f"Model {args.model_name} loaded. Starting server...")

    uvicorn.run(app, host="0.0.0.0", port=8001)
