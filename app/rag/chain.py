"""LCEL RAG chain: retriever → document_formatter → prompt → LLM → output_parser."""

import logging
import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from app.rag.prompt import get_rag_prompt, format_context
from app.retrieval.retriever import retrieve

load_dotenv()
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("OPENAI_MODEL", "gemini-2.5-flash")


def _retrieve_and_format(inputs: dict) -> dict:
    """Retrieve relevant chunks and format them into context.

    This combines the retrieval and formatting steps with error handling.
    """
    question = inputs["question"]
    top_k = inputs.get("top_k", 5)

    try:
        results = retrieve(question, top_k=top_k)

        # Store raw results for source extraction
        raw_sources = [
            {"source_file": r.source_file, "chunk_index": r.chunk_index}
            for r in results
        ]

        # Format context for the prompt
        chunks_for_context = [r.to_dict() for r in results]
        context = format_context(chunks_for_context)

        return {
            "context": context,
            "question": question,
            "sources": raw_sources,
        }
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return {
            "context": "Error: Could not retrieve relevant context.",
            "question": question,
            "sources": [],
        }


def _parse_output(llm_output: str, sources: list[dict]) -> dict:
    """Parse LLM output into answer text and structured sources.

    Args:
        llm_output: Raw text from the LLM.
        sources: List of source dicts from retrieval.

    Returns:
        Dict with 'answer' and 'sources' keys.
    """
    # Extract cited sources from the LLM answer text
    cited_sources = []
    for source_info in sources:
        source_file = source_info["source_file"]
        # Check if the source file is mentioned in the answer
        if source_file in llm_output or source_file.split("/")[-1] in llm_output:
            cited_sources.append(source_info)

    # If no sources were explicitly cited but we have sources, include all retrieved ones
    if not cited_sources and sources:
        cited_sources = sources

    return {
        "answer": llm_output.strip(),
        "sources": cited_sources,
    }


def get_llm() -> ChatOpenAI:
    """Initialize and return the OpenAI LLM instance."""
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=1024,
        request_timeout=30,
    )


def run_rag_chain(question: str, top_k: int = 5) -> dict:
    """Execute the full RAG chain for a given question.

    Pipeline: question → retriever → document_formatter → prompt → LLM → output_parser

    Args:
        question: User's natural language question.
        top_k: Number of chunks to retrieve.

    Returns:
        Dict with keys: answer, sources, model_used
    """
    try:
        # Step 1 & 2: Retrieve and format context
        retrieval_result = _retrieve_and_format({"question": question, "top_k": top_k})

        context = retrieval_result["context"]
        sources = retrieval_result["sources"]

        # Step 3: Build prompt
        prompt = get_rag_prompt()

        # Step 4: LLM
        llm = get_llm()

        # Step 5: Output parser
        output_parser = StrOutputParser()

        # Build LCEL chain: prompt → LLM → parser
        chain = prompt | llm | output_parser

        # Invoke the chain
        llm_output = chain.invoke({
            "context": context,
            "question": question,
        })

        # Parse output
        result = _parse_output(llm_output, sources)
        result["model_used"] = MODEL_NAME

        return result

    except Exception as e:
        logger.error("RAG chain failed: %s", e)
        return {
            "answer": f"I encountered an error while processing your question: {str(e)}",
            "sources": [],
            "model_used": MODEL_NAME,
        }
