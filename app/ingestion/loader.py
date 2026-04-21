"""Document loading and chunking for the RoboDesk ingestion pipeline."""

import logging
import os
from collections import defaultdict

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Map subdirectory names to document_type metadata
SUBDIR_TO_DOCTYPE = {
    "products": "product",
    "faq": "faq",
    "support": "support",
    "policies": "policy",
}


def load_documents(dataset_path: str) -> list[Document]:
    """Load all .txt files from the dataset directory tree.

    Args:
        dataset_path: Path to the dataset/ directory.

    Returns:
        List of loaded LangChain Document objects with source metadata.
    """
    dataset_path = os.path.abspath(dataset_path)
    logger.info("Loading documents from: %s", dataset_path)

    loader = DirectoryLoader(
        dataset_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()

    # Normalize source paths to use forward slashes and be relative
    for doc in documents:
        source = doc.metadata.get("source", "")
        # Make relative to dataset_path
        rel_source = os.path.relpath(source, dataset_path).replace("\\", "/")
        doc.metadata["source"] = rel_source

        # Determine document_type from subdirectory
        subdir = rel_source.split("/")[0] if "/" in rel_source else ""
        doc.metadata["document_type"] = SUBDIR_TO_DOCTYPE.get(subdir, "unknown")

    logger.info("Loaded %d documents from %s", len(documents), dataset_path)
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked Document objects with chunk_index metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    all_chunks: list[Document] = []
    chunks_per_source: dict[str, int] = defaultdict(int)

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("document_type", "unknown")

        chunks = splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["source"] = source
            chunk.metadata["document_type"] = doc_type

        all_chunks.extend(chunks)
        chunks_per_source[source] += len(chunks)

    # Log chunks per source
    for source, count in sorted(chunks_per_source.items()):
        logger.info("  %s: %d chunks", source, count)

    logger.info("Total chunks produced: %d", len(all_chunks))
    return all_chunks
