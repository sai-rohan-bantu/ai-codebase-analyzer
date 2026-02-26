import os
import json
import requests
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv

# Load env BEFORE anything else
load_dotenv()


class APICopilotReasoningEngine:
    """
    Stable OpenRouter Copilot Engine (Production + RAG Optimized)

    Fixes:
    - 401 Missing Authentication
    - 404 model routing errors
    - Deprecated model issues
    - Streaming SSE parsing bugs
    """

    def __init__(self):
        # 🔥 Strip prevents newline/space header bugs (Windows common)
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set in .env\n"
                "Add: OPENROUTER_API_KEY=sk-or-v1-xxxxx"
            )

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        # 🧠 VERY STABLE free models on OpenRouter (non-deprecated)
        self.primary_model = "deepseek/deepseek-chat"
        self.fallback_model = "mistralai/mistral-7b-instruct"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Required for OpenRouter ranking & routing
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI-Codebase-Analyzer"
        }

        print("🚀 OpenRouter Copilot Initialized")
        print("🧠 Primary Model:", self.primary_model)
        print("🔑 Key Loaded:", repr(self.api_key[:10] + "..."))

    # =====================================================
    # MAIN RESPONSE (NON-STREAM)
    # =====================================================
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        repo_name: str
    ) -> Dict[str, Any]:

        if not retrieved_chunks:
            return {
                "answer": "No relevant context found in the indexed repository.",
                "grounded_files": [],
                "context_used": 0
            }

        # Limit context for stability + cost
        selected_chunks = retrieved_chunks[:5]
        context_text = self._format_context(selected_chunks)
        prompt = self._build_prompt(query, context_text, repo_name)

        answer = self._call_with_fallback(prompt)

        grounded_files = list({
            chunk["metadata"].get("file_name", "unknown")
            for chunk in selected_chunks
        })

        return {
            "answer": answer,
            "grounded_files": grounded_files,
            "context_used": len(selected_chunks)
        }

    # =====================================================
    # STREAMING (COPILOT STYLE)
    # =====================================================
    def stream_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        repo_name: str
    ) -> Generator[str, None, None]:

        if not retrieved_chunks:
            yield "No relevant context found in the indexed repository."
            return

        selected_chunks = retrieved_chunks[:5]
        context_text = self._format_context(selected_chunks)
        prompt = self._build_prompt(query, context_text, repo_name)

        payload = {
            "model": self.primary_model,
            "messages": [
                {"role": "system", "content": "You are a professional AI code copilot."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "stream": True,
            "max_tokens": 900
        }

        try:
            with requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=120
            ) as response:

                if response.status_code != 200:
                    yield f"\n[Streaming Error {response.status_code}]: {response.text}"
                    return

                for line in response.iter_lines():
                    if not line:
                        continue

                    decoded = line.decode("utf-8")

                    if decoded.startswith("data: "):
                        data_str = decoded.replace("data: ", "").strip()

                        if data_str == "[DONE]":
                            break

                        try:
                            chunk_json = json.loads(data_str)
                            delta = chunk_json["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            continue

        except Exception as e:
            yield f"\n[Streaming Exception]: {str(e)}"

    # =====================================================
    # FALLBACK MODEL HANDLER
    # =====================================================
    def _call_with_fallback(self, prompt: str) -> str:
        for model in [self.primary_model, self.fallback_model]:
            try:
                return self._call_llm(prompt, model)
            except Exception as e:
                print(f"⚠️ Model failed ({model}): {e}")
                continue
        return "LLM failed on all available models."

    def _call_llm(self, prompt: str, model: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a professional AI code copilot."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 900
        }

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        blocks = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            content = chunk.get("content", "")
            blocks.append(f"[CHUNK {i}] File: {meta.get('file_name')}\n{content}")
        return "\n\n".join(blocks)

    def _build_prompt(self, query: str, context: str, repo_name: str) -> str:
        return f"""
You are an expert AI Codebase Analyzer and Copilot.

Repository: {repo_name}

STRICT RULES:
- Use ONLY the provided repository context
- Do NOT hallucinate missing files
- Be precise and technical
- Explain architecture, design patterns, and flow clearly

USER QUERY:
{query}

REPOSITORY CONTEXT:
{context}
"""