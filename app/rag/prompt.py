"""Prompt templates for the RoboDesk RAG chain."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """\
You are RoboDesk, an AI-powered Customer Service Desk assistant for Axiom Robotics.

Your job is to answer customer questions accurately and helpfully using ONLY the provided context.

## Rules:
1. Answer ONLY based on the provided context below. Do NOT use any external knowledge.
2. If the context does not contain enough information to answer the question, respond with: \
"I don't have enough information in my knowledge base to answer this question. \
Please contact Axiom Robotics support for further assistance."
3. Be concise, professional, and friendly.
4. At the end of your answer, list the source files you used under a "Sources:" heading.
5. If multiple context chunks are relevant, synthesize information from all of them.

## Context:
{context}
"""

HUMAN_PROMPT = "{question}"


def get_rag_prompt() -> ChatPromptTemplate:
    """Return the ChatPromptTemplate for RAG generation.

    The template expects two variables:
    - context: Formatted numbered context block from retrieved chunks
    - question: The user's question
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt.

    Args:
        chunks: List of dicts with keys: chunk_text, source_file, chunk_index

    Returns:
        Formatted context string with numbered chunks and source metadata.
    """
    if not chunks:
        return "No relevant context was found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_file", "unknown")
        chunk_idx = chunk.get("chunk_index", "?")
        text = chunk.get("chunk_text", "")
        context_parts.append(
            f"[{i}] Source: {source} (chunk {chunk_idx})\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)
