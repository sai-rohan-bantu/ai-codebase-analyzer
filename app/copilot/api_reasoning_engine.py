import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class APICopilotReasoningEngine:
    """
    Production API-Based Copilot Reasoning Engine (OpenRouter)

    Designed specifically for:
    - RAG pipelines
    - Codebase analysis
    - Repo-agnostic systems
    - FastAPI deployment
    """

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "deepseek/deepseek-r1:free"  # Free + good for code reasoning

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env file")

    # =====================================================
    # MAIN ENTRY (STRICT INPUT / OUTPUT CONTRACT)
    # =====================================================
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        repo_name: str
    ) -> Dict[str, Any]:
        """
        Structured response generator for production APIs.

        Returns:
        {
            answer: str,
            grounded_files: list,
            context_used: int
        }
        """

        if not retrieved_chunks:
            return {
                "answer": "No relevant context found in the indexed repository.",
                "grounded_files": [],
                "context_used": 0
            }

        context_text = self._format_context(retrieved_chunks)
        prompt = self._build_prompt(query, context_text, repo_name)
        answer = self._call_llm(prompt)

        grounded_files = list({
            chunk["metadata"].get("file_name", "unknown")
            for chunk in retrieved_chunks
        })

        return {
            "answer": answer,
            "grounded_files": grounded_files,
            "context_used": len(retrieved_chunks)
        }

    # =====================================================
    # CONTEXT FORMATTER (MATCHES YOUR METADATA STRUCTURE)
    # =====================================================
    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Formats chunks using YOUR existing metadata:
        file_path, file_name, language, start_line, end_line, chunk_type
        """
        formatted_blocks = []

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            content = chunk.get("content", "")

            block = f"""
[CHUNK {i}]
File: {metadata.get("file_name")}
Path: {metadata.get("file_path")}
Language: {metadata.get("language")}
Chunk Type: {metadata.get("chunk_type")}
Lines: {metadata.get("start_line", "N/A")} - {metadata.get("end_line", "N/A")}

Code:
{content}
"""
            formatted_blocks.append(block)

        return "\n".join(formatted_blocks)

    # =====================================================
    # GROUNDED PROMPT (COPILOT STYLE)
    # =====================================================
    def _build_prompt(self, query: str, context: str, repo_name: str) -> str:
        """
        Repo-agnostic, anti-hallucination prompt.
        """
        return f"""
You are an expert AI Codebase Analyzer and Copilot.

Repository: {repo_name}

STRICT RULES:
- Use ONLY the provided repository context
- Do NOT hallucinate missing files or code
- Be precise and technical
- Support multi-language codebases (Java, Python, JS, HTML, etc.)
- Explain architecture, design patterns, and flow clearly

USER QUERY:
{query}

REPOSITORY CONTEXT:
{context}

INSTRUCTIONS:
1. Identify relevant files/classes
2. Explain the implementation clearly
3. Reference actual file names when possible
4. If context is insufficient, say it explicitly
"""

    # =====================================================
    # OPENROUTER API CALL (DEPLOYMENT SAFE)
    # =====================================================
    def _call_llm(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "AI-Codebase-Analyzer"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional code copilot."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # LOW = better RAG accuracy
            "max_tokens": 900
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            return "LLM request timed out."
        except requests.exceptions.RequestException as e:
            return f"OpenRouter API error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    # =====================================================
    # STREAMING RESPONSE (COPILOT STYLE)
    # =====================================================
    def stream_response(
        self,
        query: str,
        retrieved_chunks: list,
        repo_name: str
    ):
        """
        Streaming LLM response (Copilot-style).
        Yields tokens progressively.
        """
        if not retrieved_chunks:
            yield "No relevant context found in the indexed repository."
            return

        context_text = self._format_context(retrieved_chunks)
        prompt = self._build_prompt(query, context_text, repo_name)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "AI-Codebase-Analyzer"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional AI code copilot."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "stream": True  # 🔥 Enables streaming
        }

        try:
            with requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=120
            ) as response:

                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")

                        # OpenRouter streaming format
                        if decoded.startswith("data: "):
                            data_str = decoded.replace("data: ", "").strip()

                            if data_str == "[DONE]":
                                break

                            try:
                                import json
                                json_data = json.loads(data_str)
                                delta = json_data["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield delta
                            except Exception:
                                continue

        except Exception as e:
            yield f"\n[Streaming Error]: {str(e)}"