"""Retrieval service: vector similarity search against the Chroma vector store."""

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from app.ingestion.embedder import get_embedding_model
from app.ingestion.indexer import get_vectorstore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with chunk text, metadata, and similarity score."""

    chunk_text: str
    source_file: str
    chunk_index: int
    similarity_score: float

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "chunk_text": self.chunk_text,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "similarity_score": round(self.similarity_score, 4),
        }


def retrieve(query: str, top_k: int = 5) -> list[RetrievalResult]:
    """Perform similarity search against the indexed vector store.

    Args:
        query: Natural-language query string.
        top_k: Number of top results to return (default: 5).

    Returns:
        List of RetrievalResult objects sorted by similarity score (highest first).
        Returns an empty list with a logged message if no results are found.
    """
    embeddings = get_embedding_model()
    vectorstore = get_vectorstore(embeddings)

    # Perform similarity search with scores
    results: list[tuple[Document, float]] = vectorstore.similarity_search_with_relevance_scores(
        query, k=top_k
    )

    if not results:
        logger.warning("No results found for query: '%s'", query)
        return []

    retrieval_results = []
    for doc, score in results:
        retrieval_results.append(
            RetrievalResult(
                chunk_text=doc.page_content,
                source_file=doc.metadata.get("source", "unknown"),
                chunk_index=doc.metadata.get("chunk_index", -1),
                similarity_score=score,
            )
        )

    logger.info("Retrieved %d results for query: '%s'", len(retrieval_results), query[:80])
    return retrieval_results
