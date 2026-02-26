from dotenv import load_dotenv
load_dotenv()  # MUST be first (loads environment variables BEFORE router import)

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router

# 🔍 Generic debug (works for any provider)
print("DEBUG OPENROUTER KEY:", repr(os.getenv("OPENROUTER_API_KEY")))
print("DEBUG GROQ KEY:", repr(os.getenv("GROQ_API_KEY")))
print("DEBUG OPENAI KEY:", repr(os.getenv("OPENAI_API_KEY")))

app = FastAPI(
    title="AI Codebase Analyzer Copilot",
    version="2.0.0"
)

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ai-codebase-analyzer",
        "env_loaded": True
    }

# =========================
# CORS (for UI / frontend)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# API ROUTES (RAG + Copilot)
# =========================
app.include_router(router, prefix="/api")

# =========================
# SERVE UI AT ROOT "/"
# =========================
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")