"""
Advanced Multi-Language Semantic Code Chunker (Production Grade)

Features:
- Preserves original ingestion metadata
- AST-based chunking for code
- Fallback chunking for docs/config
- Multi-language balanced (Java, Python, JS, HTML, MD)
- Copilot-style semantic chunking
"""

from typing import List, Dict
from .ast_parser import ASTParser


# Node types we care about per language (Tree-sitter)
SEMANTIC_NODE_TYPES = {
    "python": ["function_definition", "class_definition"],
    "java": ["class_declaration", "method_declaration", "interface_declaration"],
    "javascript": ["function_declaration", "class_declaration", "method_definition"],
    "typescript": ["function_declaration", "class_declaration", "method_definition"],
    "react": ["function_declaration", "class_declaration"],
}


class CodeChunker:
    def __init__(self, fallback_chunk_size: int = 400):
        self.ast_parser = ASTParser()
        self.fallback_chunk_size = fallback_chunk_size

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Main entry point.
        Converts file-level documents → semantic chunks.
        """
        all_chunks = []

        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            language = metadata.get("language", "unknown")

           # print(f"[DEBUG] File: {metadata.get('file_name')} | Language: {language}") 

            # Try AST-based chunking first
            tree = self.ast_parser.get_tree(content, language)

            #print(f"[DEBUG] AST Tree: {tree is not None}")

            if tree:
                chunks = self._chunk_via_ast(
                    tree=tree,
                    source_code=content,
                    base_metadata=metadata,
                    language=language,
                )
            else:
                # Fallback for markdown, css, json, yaml, etc.
                chunks = self._fallback_chunking(content, metadata)

            all_chunks.extend(chunks)

        print(f"[INFO] Total semantic chunks created: {len(all_chunks)}")
        return all_chunks

    def _chunk_via_ast(
        self,
        tree,
        source_code: str,
        base_metadata: Dict,
        language: str,
    ) -> List[Dict]:
        """
        Recursive AST traversal for deep semantic extraction.
        Fixes:
        - Java nested classes
        - Python nested functions
        - Deep AST structures (production requirement)
        """
        chunks = []
        target_nodes = SEMANTIC_NODE_TYPES.get(language, [])

        def traverse(node):
            # Check if this node is a meaningful semantic unit
            if node.type in target_nodes:
                chunk_text = source_code[node.start_byte: node.end_byte].strip()

                if chunk_text:
                    enriched_metadata = {
                        **base_metadata,  # 🔥 Preserve ingestion metadata
                        "chunk_type": node.type,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "chunking_strategy": "ast_semantic",
                    }

                    chunks.append(
                        {
                            "content": chunk_text,
                            "metadata": enriched_metadata,
                        }
                    )

            # 🔥 CRITICAL: Recursive traversal (handles nested AST)
            for child in node.children:
                traverse(child)

        # Start traversal from root node
        traverse(tree.root_node)

        # If no semantic nodes found, fallback to text chunking
        if not chunks:
            return self._fallback_chunking(source_code, base_metadata)

        return chunks

    def _fallback_chunking(
        self,
        content: str,
        base_metadata: Dict,
    ) -> List[Dict]:
        """
        Fallback chunking for:
        - Markdown
        - HTML (if AST not useful)
        - CSS
        - JSON/YAML
        - Small or unsupported files
        """
        chunks = []
        start = 0
        text_length = len(content)

        while start < text_length:
            end = start + self.fallback_chunk_size
            chunk_text = content[start:end].strip()

            if chunk_text:
                enriched_metadata = {
                    **base_metadata,  # Preserve original metadata
                    "chunk_type": "text_chunk",
                    "chunking_strategy": "fallback_text",
                }

                chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": enriched_metadata,
                    }
                )

            start += self.fallback_chunk_size

        return chunks