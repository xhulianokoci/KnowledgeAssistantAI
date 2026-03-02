"""
LLMOPS - OBSERVABILITY & RELIABILITY
======================================
LLMOps = applying DevOps practices to LLM applications.

This module covers:
- Logging every LLM call (inputs, outputs, latency, tokens)
- Retry logic (automatic retries on failure)
- Simple evaluation (is the answer grounded in the retrieved docs?)
- Session tracking

In production, we would send these logs to LangSmith, LangFuse, or similar.
Here we save to a local JSONL file so you can see what's happening.
"""

import json
import time
import functools
from datetime import datetime
from pathlib import Path


LOG_FILE = "llmops_log.jsonl"


def log_llm_call(
    query: str,
    retrieved_chunks: list,
    answer: str,
    model: str,
    latency_ms: float,
    mode: str = "rag"
):
    """
    Logs every LLM interaction to a JSONL file.
    JSONL = one JSON object per line, easy to parse and analyze.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "model": model,
        "query": query,
        "query_length": len(query),
        "chunks_retrieved": len(retrieved_chunks),
        "answer_length": len(answer),
        "latency_ms": round(latency_ms, 2),
        # Simple quality check: does the answer contain words from retrieved docs?
        "grounding_score": _compute_grounding_score(answer, retrieved_chunks),
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_entry


def _compute_grounding_score(answer: str, chunks: list) -> float:
    """
    Simple grounding check: what fraction of answer words appear in retrieved chunks?
    Score 0.0 (no grounding) to 1.0 (fully grounded).
    This is a naive eval — production uses LLM-based evaluation.
    """
    if not chunks or not answer:
        return 0.0

    answer_words = set(answer.lower().split())
    chunk_text = " ".join([
        c.page_content if hasattr(c, "page_content") else str(c)
        for c in chunks
    ]).lower()
    chunk_words = set(chunk_text.split())

    # Remove common stop words for a better signal
    stop_words = {"the", "a", "an", "is", "it", "in", "of", "to", "and", "or", "for",
                  "that", "this", "with", "on", "are", "was", "be", "have", "has"}
    meaningful_answer_words = answer_words - stop_words

    if not meaningful_answer_words:
        return 0.0

    overlap = meaningful_answer_words.intersection(chunk_words)
    return round(len(overlap) / len(meaningful_answer_words), 2)


def with_retry(max_retries: int = 3, delay_seconds: float = 1.0):
    """
    Decorator that adds automatic retry logic to any function.
    If the LLM call fails, it waits and tries again.

    Usage:
        @with_retry(max_retries=3)
        def call_llm(...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        print(f"⚠️  Attempt {attempt} failed: {e}. Retrying in {delay_seconds}s...")
                        time.sleep(delay_seconds)
                    else:
                        print(f"❌ All {max_retries} attempts failed.")
            raise last_error
        return wrapper
    return decorator


def get_session_stats() -> dict:
    """
    Reads the log file and computes session statistics.
    Shows total calls, average latency, and average grounding score.
    """
    if not Path(LOG_FILE).exists():
        return {"message": "No logs yet. Ask a question first!"}

    logs = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))

    if not logs:
        return {"message": "Log file is empty."}

    latencies = [l["latency_ms"] for l in logs]
    grounding = [l["grounding_score"] for l in logs]

    return {
        "total_queries": len(logs),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "avg_grounding_score": round(sum(grounding) / len(grounding), 2),
        "last_query": logs[-1]["query"] if logs else None,
        "log_file": LOG_FILE
    }


def get_recent_logs(n: int = 5) -> list:
    """Returns the N most recent log entries."""
    if not Path(LOG_FILE).exists():
        return []

    logs = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))

    return logs[-n:]
