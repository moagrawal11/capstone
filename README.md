**Summary of How to Run It**

Once: uv run python convert_pdfs.py (To get the raw text)
Once: uv run python ingest.py (Runs Phase 1)
Always: uv run uvicorn app.main:app --reload --port 8000 (Starts the FastAPI server)