# Architecture notes

## Structured multimodal chunk metadata

Each image, table and equation chunk written by `RAGAnything` is stored in LightRAG's `text_chunks` KV together with structured fields built by `raganything.utils.build_modal_chunk_metadata`. Missing values are `None` (never invented):

| Field | Meaning |
|---|---|
| `is_multimodal` | always `True` for these chunks |
| `original_type` | `image`, `table` or `equation` |
| `page_idx` | page of the source item (defaults to `item["page_idx"]`, else `0`) |
| `asset_key` | object-store key of the item's image, if the caller set `item["asset_key"]` |
| `table_body` | raw table content (tables only), before prompt formatting |
| `caption` | image/table captions of the item joined with `", "` |

### `asset_key` in `content_list` and chunk templates

A caller (for example the `ami_rag` worker) may upload the extracted image to an object store and set `item["asset_key"]` on the `content_list` item before `insert_content_list`. When present, the image/table chunk templates print `Asset: <asset_key>` (`raganything.utils.format_asset_line`) instead of the temporary local image path, so the chunk keeps a durable reference. Items without `asset_key` behave as before.

## AMI service

The `ami_rag/` package (ingest worker + `POST /v2/rag` API built on `RAGAnything`) is documented in [ami_service.md](ami_service.md) (Vietnamese).
