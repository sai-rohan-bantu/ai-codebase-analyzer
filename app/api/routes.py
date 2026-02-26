from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uuid
from fastapi.responses import StreamingResponse
from app.ingestion.pipeline import IngestionPipeline
from app.vectorstore.index_manager import IndexManager
from app.rag.pipeline import CodebaseRAGPipeline
from app.copilot.api_reasoning_engine import APICopilotReasoningEngine
from app.embedding.embedder import LocalEmbedder
router = APIRouter()

# Singleton services (production pattern)
ingestion_pipeline = IngestionPipeline()
index_manager = IndexManager()
copilot_engine = APICopilotReasoningEngine()
embedder = LocalEmbedder()
# In-memory repo session store (can upgrade to Redis later)
# ACTIVE_REPOS = {}
import json
import os

SESSION_FILE = "repo_sessions.json"

def load_sessions():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    return {}

def save_sessions():
    with open(SESSION_FILE, "w") as f:
        json.dump(ACTIVE_REPOS, f, indent=2)

ACTIVE_REPOS = load_sessions()

# =========================
# Request Schemas
# =========================
class RepoRequest(BaseModel):
    repo_url: str


class ChatRequest(BaseModel):
    repo_id: str
    query: str

# =========================
# BACKGROUND INGESTION TASK
# =========================
def background_ingest(repo_url: str, repo_id: str):
    try:
        ACTIVE_REPOS[repo_id]["status"] = "indexing"

        # Step 1: Run your existing ingestion pipeline
        documents = ingestion_pipeline.ingest_from_github(repo_url)

        repo_name = repo_url.split("/")[-1]

        # Step 2: Chunking already inside pipeline OR handled downstream
        chunks = documents  # if your pipeline already outputs chunks
        # If pipeline outputs documents, keep your existing chunker logic here

        # Step 3: Build or load index (SMART CACHE)
        index_manager.load_or_build_index(repo_name, chunks)

        ACTIVE_REPOS[repo_id]["status"] = "ready"
        ACTIVE_REPOS[repo_id]["repo_name"] = repo_name
        save_sessions()

    except Exception as e:
        ACTIVE_REPOS[repo_id]["status"] = "failed"
        ACTIVE_REPOS[repo_id]["error"] = str(e)


# =========================
# 1. UPLOAD REPO (NON-BLOCKING)
# =========================
@router.post("/ingest")
def ingest_repository(request: RepoRequest, background_tasks: BackgroundTasks):
    """
    User uploads GitHub repo.
    Starts ingestion in background (NON-BLOCKING).
    """
    repo_id = f"repo_{uuid.uuid4().hex[:8]}"

    save_sessions()

    ACTIVE_REPOS[repo_id] = {
        "status": "queued",
        "repo_url": request.repo_url
    }

    # Run ingestion asynchronously
    background_tasks.add_task(background_ingest, request.repo_url, repo_id)

    return {
        "repo_id": repo_id,
        "status": "indexing_started",
        "message": "Repository ingestion started in background. You can start chatting now."
    }


# =========================
# 2. CHECK INGESTION STATUS
# =========================
@router.get("/status/{repo_id}")
def get_repo_status(repo_id: str):
    if repo_id not in ACTIVE_REPOS:
        raise HTTPException(status_code=404, detail="Repo session not found")

    return ACTIVE_REPOS[repo_id]


# =========================
# 3. COPILOT CHAT (WORKS EVEN DURING INDEXING)
# =========================
@router.post("/chat")
def chat_with_repo(request: ChatRequest):
    repo_id = request.repo_id
    query = request.query

    if repo_id not in ACTIVE_REPOS:
        raise HTTPException(status_code=404, detail="Invalid repo_id. Please ingest repo first.")

    repo_info = ACTIVE_REPOS[repo_id]
    repo_name = repo_info.get("repo_url", "").split("/")[-1]

    try:
        # Load index (will use cached index if exists)
        vector_store = index_manager.load_or_build_index(repo_name, [])

        # Run RAG retrieval
        rag_pipeline = CodebaseRAGPipeline(vector_store=vector_store,embedder=embedder)
        retrieved_chunks = rag_pipeline.get_context(query)

        # Copilot reasoning (OpenRouter API)
        ai_response = copilot_engine.generate_response(
            query=query,
            retrieved_chunks=retrieved_chunks,
            repo_name=repo_name
        )

        return {
            "repo_id": repo_id,
            "repo_status": repo_info["status"],  # indexing or ready
            "response": ai_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
def stream_chat(request: ChatRequest):
    """
    REAL Copilot-style streaming (token-by-token)
    Low latency + flush-friendly + UI compatible
    """
    repo_id = request.repo_id
    query = request.query

    if repo_id not in ACTIVE_REPOS:
        raise HTTPException(status_code=404, detail="Invalid repo_id. Please ingest repo first.")

    repo_info = ACTIVE_REPOS[repo_id]
    repo_name = repo_info.get("repo_url", "").split("/")[-1]

    try:
        # Load vector store (cached index)
        vector_store = index_manager.load_or_build_index(repo_name, [])

        # RAG Retrieval
        rag_pipeline = CodebaseRAGPipeline(
            vector_store=vector_store,
            embedder=embedder
        )
        retrieved_chunks = rag_pipeline.get_context(query)

        # 🔥 TRUE streaming generator (with flush-friendly chunks)
        def token_generator():
            try:
                for token in copilot_engine.stream_response(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    repo_name=repo_name
                ):
                    # Important: small chunks = smoother streaming
                    yield token
            except Exception as e:
                yield f"\n[Streaming Error]: {str(e)}"

        return StreamingResponse(
            token_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# @router.post("/chat/stream")
# def stream_chat(request: ChatRequest):
#     """
#     Copilot-style streaming chat endpoint.
#     Works even during indexing.
#     """
#     repo_id = request.repo_id
#     query = request.query

#     if repo_id not in ACTIVE_REPOS:
#         raise HTTPException(status_code=404, detail="Invalid repo_id. Please ingest repo first.")

#     repo_info = ACTIVE_REPOS[repo_id]
#     repo_name = repo_info.get("repo_url", "").split("/")[-1]

#     try:
#         # Load vector store (cached index)
#         vector_store = index_manager.load_or_build_index(repo_name, [])

#         # RAG Retrieval
#         rag_pipeline = CodebaseRAGPipeline(vector_store=vector_store,embedder=embedder)
#         retrieved_chunks = rag_pipeline.get_context(query)

#         # Streaming generator
#         def token_generator():
#             for token in copilot_engine.stream_response(
#                 query=query,
#                 retrieved_chunks=retrieved_chunks,
#                 repo_name=repo_name
#             ):
#                 yield token

#         return StreamingResponse(
#             token_generator(),
#             media_type="text/plain"
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))