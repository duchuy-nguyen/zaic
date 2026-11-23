def agent_view(row: dict) -> dict:
    """Gold-text train mode only. Deliberate allowlist excludes label/final_state."""
    if "audio_text" not in row:
        raise ValueError("This runner is gold-text TRAIN-only; public/private need an ASR provider")
    history = []
    for turn in row.get("history", []):
        if turn["role"] == "user":
            if "audio_text" not in turn:
                raise ValueError("Missing gold history transcript; no silent fallback")
            history.append({"role": "user", "text": turn["audio_text"]})
        else:
            history.append({k: turn[k] for k in ("role", "text", "tool_calls", "tool_results") if k in turn})
    return {"id": row["id"], "latest_text": row["audio_text"], "context": row["context"], "history": history}
