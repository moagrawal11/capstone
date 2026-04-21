"""Standalone ingestion runner — runs the full pipeline end-to-end.

Usage:
    uv run python ingest.py [--dataset-path DATASET_PATH]
"""

import argparse
import logging
import time

from app.ingestion.loader import load_documents, chunk_documents
from app.ingestion.embedder import get_embedding_model
from app.ingestion.indexer import index_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ingestion(dataset_path: str = "dataset") -> dict:
    start = time.time()

    print("=" * 60)
    print("  RoboDesk — Document Ingestion Pipeline")
    print("=" * 60)
    print()

    # Step 1: Load documents
    print("[1/3] Loading documents...")
    documents = load_documents(dataset_path)
    files_loaded = len(documents)
    print(f"   Loaded {files_loaded} documents")
    print()

    # Step 2: Chunk documents
    print("[2/3] Chunking documents...")
    chunks = chunk_documents(documents)
    chunks_created = len(chunks)
    print(f"   Created {chunks_created} chunks")
    print()

    # Step 3: Generate embeddings & index
    print("[3/3] Generating embeddings & indexing into Chroma...")
    embeddings = get_embedding_model()
    chunks_indexed = index_chunks(chunks, embeddings)
    print(f"   Indexed {chunks_indexed} chunks")
    print()

    duration = round(time.time() - start, 2)

    # Summary report
    print("=" * 60)
    print("  Ingestion Summary")
    print("=" * 60)
    print(f"  Files loaded:     {files_loaded}")
    print(f"  Chunks created:   {chunks_created}")
    print(f"  Chunks indexed:   {chunks_indexed}")
    print(f"  Time taken:       {duration}s")
    print("=" * 60)

    return {
        "files_loaded": files_loaded,
        "chunks_created": chunks_created,
        "chunks_indexed": chunks_indexed,
        "duration_seconds": duration,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoboDesk Document Ingestion Pipeline")
    parser.add_argument(
        "--dataset-path",
        default="dataset",
        help="Path to the dataset directory (default: dataset)",
    )
    args = parser.parse_args()

    run_ingestion(args.dataset_path)
