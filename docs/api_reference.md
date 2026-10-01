# API reference (additions)

## `RAGAnything.aquery_data` / `query_data`

```python
result = await rag.aquery_data(query: str, mode: str = "mix", **param_kwargs)
result = rag.query_data(query, mode="mix", **param_kwargs)   # synchronous wrapper
```

Retrieves structured data without generating an LLM answer. `**param_kwargs` are passed to LightRAG's `QueryParam` (for example `top_k`, `chunk_top_k`, `enable_rerank`). Raises `RuntimeError` if LightRAG cannot be initialized.

Returns the same dict as `LightRAG.aquery_data`: `status`, `message`, `data` (`entities`, `relationships`, `chunks`, `references`) and `metadata`. Each item in `data["chunks"]` is additionally enriched from the `text_chunks` KV (failures are logged, never raised):

| Field | Value |
|---|---|
| `modality` | `"text"`, or the chunk's `original_type` (`image` / `table` / `equation`) for multimodal chunks |
| `asset_key` | object-store key of the image, or `None` |
| `table_body` | raw table content, or `None` |
| `page_idx` | source page, or `None` |
| `caption` | joined captions, or `None` |

Chunk metadata and the `item["asset_key"]` convention are described in [architecture.md](architecture.md). Service-level usage (`POST /v2/rag`) is in [ami_service.md](ami_service.md).
