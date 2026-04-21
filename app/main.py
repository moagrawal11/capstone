"""FastAPI application entry point for RoboDesk."""

import logging

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.routes import router

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(
    title="RoboDesk API",
    description=(
        "AI-powered Customer Service Desk assistant for Axiom Robotics. "
        "Provides RAG-based question answering over the RoboDesk knowledge base."
    ),
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():
    """Root endpoint returning service info."""
    return {
        "service": "RoboDesk API",
        "version": "1.0.0",
        "docs": "/docs",
    }
