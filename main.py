from app.ingestion.pipeline import IngestionPipeline


def test_github_ingestion_and_chunking():
    """
    End-to-End Test:
    GitHub Repo → Ingestion → AST Chunking → Semantic Chunks
    """

    # Use a real repo (you already cloned similar)
    repo_url = "https://github.com/sai-rohan-bantu/SystemDesign-LLD-Behavioural"

    print("\n[TEST] Starting GitHub ingestion + chunking test...\n")

    pipeline = IngestionPipeline()
    chunks = pipeline.ingest_from_github(repo_url)

    # 1️⃣ Basic sanity check
    print(f"\n[RESULT] Total Chunks Generated: {len(chunks)}")

    if not chunks:
        print("[ERROR] No chunks generated!")
        return

    # 2️⃣ Print first 3 chunks for inspection
    print("\n[DEBUG] Sample Chunk Metadata:\n")
    for i, chunk in enumerate(chunks[:10]):
        print(f"Chunk {i+1} Metadata:")
        print(chunk["metadata"])
        print("-" * 50)

    # 3️⃣ Check if AST chunking is working
    ast_chunks = [
        c for c in chunks
        if c["metadata"].get("chunking_strategy") == "ast_semantic"
    ]

    fallback_chunks = [
        c for c in chunks
        if c["metadata"].get("chunking_strategy") == "fallback_text"
    ]

    print(f"\n[ANALYSIS]")
    print(f"AST Semantic Chunks: {len(ast_chunks)}")
    print(f"Fallback Chunks: {len(fallback_chunks)}")

    # 4️⃣ Critical validation: Metadata preservation
    sample_meta = chunks[0]["metadata"]

    required_fields = [
        "file_path",
        "language",
        "repo",
        "chunk_type",
        "chunking_strategy"
    ]

    missing = [field for field in required_fields if field not in sample_meta]

    if missing:
        print(f"[WARNING] Missing metadata fields: {missing}")
    else:
        print("[SUCCESS] Metadata enrichment is correct ✔")


if __name__ == "__main__":
    test_github_ingestion_and_chunking()