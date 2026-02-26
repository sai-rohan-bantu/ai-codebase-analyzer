from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router

app = FastAPI(
    title="AI Codebase Analyzer Copilot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Backend APIs
app.include_router(router, prefix="/api")

# Serve UI automatically at root
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")




# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# # Import your API routes (we created earlier)
# from app.api.routes import router
# from fastapi.staticfiles import StaticFiles
# import webbrowser
# import threading

# app = FastAPI(
#     title="AI Codebase Analyzer Copilot",
#     description="Repo-Agnostic AI Codebase Analyzer with RAG + Streaming + OpenRouter",
#     version="1.0.0"
# )

# def open_browser():
#     webbrowser.open("http://127.0.0.1:8000")

# @app.on_event("startup")
# def startup_event():
#     threading.Timer(1.5, open_browser).start()

# # Enable CORS (important for future UI like React)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Register all API endpoints
# app.include_router(router, prefix="/api")

# app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

# @app.get("/")
# def root():
#     return {
#         "message": "AI Codebase Copilot API is running 🚀",
#         "features": [
#             "Async Repo Ingestion",
#             "Streaming Copilot Chat",
#             "RAG + FAISS Indexing",
#             "OpenRouter LLM Integration"
#         ]
#     }


# @app.get("/health")
# def health_check():
#     return {
#         "status": "healthy",
#         "service": "ai-codebase-analyzer"
#     }