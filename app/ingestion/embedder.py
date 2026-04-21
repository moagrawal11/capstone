"""Embedding model wrapper for the RoboDesk ingestion pipeline."""

import logging

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Initialize and return the HuggingFace embedding model.

    Returns:
        HuggingFaceEmbeddings instance using all-MiniLM-L6-v2.
    """
    logger.info("Loading embedding model: %s", MODEL_NAME)

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Log embedding dimensions by generating a test embedding
    test_embedding = embeddings.embed_query("test")
    logger.info("Embedding dimensions: %d", len(test_embedding))

    return embeddings
