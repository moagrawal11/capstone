"""Vector store indexing (Chroma) for the RoboDesk ingestion pipeline."""

import hashlib
import logging
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "roboDesk-kb"
PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chroma_db")
PERSIST_DIR = os.path.normpath(PERSIST_DIR)


def _generate_chunk_id(chunk: Document) -> str:
    """Generate a deterministic ID for a chunk based on source and chunk_index.

    This ensures re-running ingestion does not create duplicate entries.
    """
    source = chunk.metadata.get("source", "unknown")
    chunk_index = chunk.metadata.get("chunk_index", 0)
    key = f"{source}::{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()


def index_chunks(
    chunks: list[Document],
    embeddings: Embeddings,
    persist_directory: str | None = None,
) -> int:
    """Upsert chunks into the Chroma vector store.

    Args:
        chunks: List of chunked Document objects with metadata.
        embeddings: Embedding model instance.
        persist_directory: Optional override for the Chroma persist directory.

    Returns:
        Number of chunks indexed in the collection.
    """
    persist_dir = persist_directory or PERSIST_DIR
    os.makedirs(persist_dir, exist_ok=True)

    logger.info("Indexing %d chunks into Chroma collection '%s'", len(chunks), COLLECTION_NAME)
    logger.info("Persist directory: %s", persist_dir)

    # Generate deterministic IDs for upsert (deduplication)
    chunk_ids = [_generate_chunk_id(chunk) for chunk in chunks]

    # Create / connect to Chroma and upsert
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
        ids=chunk_ids,
    )

    # Verify indexed count
    collection_count = vectorstore._collection.count()
    logger.info("Chroma collection '%s' now contains %d documents", COLLECTION_NAME, collection_count)

    if collection_count != len(chunks):
        logger.warning(
            "Count mismatch: expected %d, got %d (may be due to prior data)",
            len(chunks),
            collection_count,
        )

    return collection_count


def get_vectorstore(
    embeddings: Embeddings,
    persist_directory: str | None = None,
) -> Chroma:
    """Connect to the existing Chroma vector store.

    Args:
        embeddings: Embedding model instance.
        persist_directory: Optional override for the Chroma persist directory.

    Returns:
        Chroma vector store instance.
    """
    persist_dir = persist_directory or PERSIST_DIR
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
