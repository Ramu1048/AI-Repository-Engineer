# AI Repository Engineer — Member 5 Frontend & Integration

This module implements the **FastAPI Backend**, **Streamlit Frontend**, **Docker deployment setup**, and **Integration Testing** for the AI Repository Engineer codebase.

---

## Architecture Overview

```
                  ┌──────────────────┐
                  │   Streamlit UI   │ (app.py)
                  │   (Port 8501)    │
                  └────────┬─────────┘
                           │ HTTP REST
                           ▼
                  ┌──────────────────┐
                  │   FastAPI API    │ (backend/api.py)
                  │   (Port 8000)    │
                  └────────┬─────────┘
                           │ Python SDKs / functions
                           ▼
           ┌──────────────────────────────┐
           │   Repository Analysis & RAG  │ (Stubs for Members 1-4)
           │   (Ingestion, Chunks, QA)    │
           └──────────────────────────────┘
```

---

## Local Setup Instructions

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional, for containerised run)

### Running Locally with Pip
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the FastAPI Backend:
   ```bash
   uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
   ```
3. Run the Streamlit UI:
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```
4. Access:
   - FastAPI docs: `http://localhost:8000/docs`
   - Streamlit UI: `http://localhost:8501`

### Running with Docker Compose
1. Ensure your `.env` is configured (copy from `.env.example`).
2. Run command:
   ```bash
   docker-compose up --build
   ```
3. Access Streamlit UI at `http://localhost:8501`.

---

## Running Integration Tests
To run the automated endpoint validation and integration tests, use:
```bash
pytest tests/ -v
```
