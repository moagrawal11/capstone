"""FastAPI route handlers for the RoboDesk API."""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ingestion.loader import load_documents, chunk_documents
from app.ingestion.embedder import get_embedding_model
from app.ingestion.indexer import index_chunks, get_vectorstore, COLLECTION_NAME
from app.retrieval.retriever import retrieve
from app.rag.chain import run_rag_chain

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Request / Response models ────────────────────────────────────────────────


class IngestRequest(BaseModel):
    """Optional request body for the /ingest endpoint."""
    dataset_path: str = Field(default="dataset", description="Path to the dataset directory")


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""
    files_loaded: int
    chunks_created: int
    chunks_indexed: int
    duration_seconds: float


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""
    question: str
    answer: str
    sources: list[dict]
    model_used: str
    timestamp: str


class RetrieveRequest(BaseModel):
    """Request body for the /retrieve endpoint."""
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class RetrieveResponse(BaseModel):
    """Response body for the /retrieve endpoint."""
    query: str
    results: list[dict]
    total_results: int


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    vector_store_connected: bool
    indexed_document_count: int
    collection_name: str


# ── Route handlers ───────────────────────────────────────────────────────────


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(request: IngestRequest = IngestRequest()):
    """Trigger the full ingestion pipeline on demand."""
    start = time.time()

    try:
        # Load documents
        documents = load_documents(request.dataset_path)
        files_loaded = len(documents)

        # Chunk documents
        chunks = chunk_documents(documents)
        chunks_created = len(chunks)

        # Embed and index
        embeddings = get_embedding_model()
        chunks_indexed = index_chunks(chunks, embeddings)

        duration = round(time.time() - start, 2)

        return IngestResponse(
            files_loaded=files_loaded,
            chunks_created=chunks_created,
            chunks_indexed=chunks_indexed,
            duration_seconds=duration,
        )
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """Run the full RAG chain (retrieval + LLM generation)."""
    try:
        result = run_rag_chain(request.question, top_k=request.top_k)

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            model_used=result["model_used"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(request: RetrieveRequest):
    """Run retrieval only (no LLM), returns raw chunks with scores."""
    try:
        results = retrieve(request.query, top_k=request.top_k)

        return RetrieveResponse(
            query=request.query,
            results=[r.to_dict() for r in results],
            total_results=len(results),
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
def health_endpoint():
    """Return service status, vector store connection status, and indexed document count."""
    try:
        embeddings = get_embedding_model()
        vectorstore = get_vectorstore(embeddings)
        count = vectorstore._collection.count()

        return HealthResponse(
            status="healthy",
            vector_store_connected=True,
            indexed_document_count=count,
            collection_name=COLLECTION_NAME,
        )
    except Exception as e:
        logger.warning("Health check — vector store unavailable: %s", e)
        return HealthResponse(
            status="degraded",
            vector_store_connected=False,
            indexed_document_count=0,
            collection_name=COLLECTION_NAME,
        )
