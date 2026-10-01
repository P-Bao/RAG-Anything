import httpx

from ami_rag.settings import Settings, get_settings


def build_rerank_model_func(settings: Settings | None = None):
    """Build a rerank_model_func compatible with LightRAG's apply_rerank contract.

    Wraps the external rerank service (POST {base}/rerank returning
    {scores, ranked_indices}) and translates the response into the index-based
    format LightRAG expects: [{"index": i, "relevance_score": s}, ...].
    Exceptions propagate: LightRAG catches them and falls back to original chunks.
    """
    settings = settings or get_settings()
    base_url = settings.RERANK_BASE_URL.rstrip("/")

    async def rerank_model_func(query: str, documents: list[str], top_n: int | None = None, **_):
        payload = {
            "query": query,
            "documents": documents,
            "top_k": top_n if top_n is not None else settings.RERANK_TOP_K,
        }
        async with httpx.AsyncClient(timeout=settings.RERANK_TIMEOUT) as client:
            resp = await client.post(f"{base_url}/rerank", json=payload)
            resp.raise_for_status()
            body = resp.json()

        scores = body.get("scores") or []
        indices = body.get("ranked_indices")
        if indices is None:
            indices = sorted(range(len(scores)), key=lambda i: -scores[i])
        results = []
        for i in indices:
            idx = int(i)
            if 0 <= idx < len(scores):
                results.append({"index": idx, "relevance_score": float(scores[idx])})
        return results

    return rerank_model_func
