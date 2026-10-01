from ami_rag.queue.events import RagEvent
from ami_rag.sources import source_hash
from ami_rag.workers.ingest_worker import IngestWorker

from .conftest import FakeDocsRepo, FakeRAGAnything

PDF_ID = "64b000000000000000000001"
CRAWL_ID = "64b000000000000000000003"
DOCX_ID = "64b000000000000000000004"
TEXT_ID = "64b000000000000000000005"
TXT_ID = "64b000000000000000000006"


def _worker(rag_anything, fake_docs_repo, fake_state_repo, fake_queue, asset_store, max_delivery=3):
    class _S:
        WORKER_MAX_DELIVERY = max_delivery
        WORKER_BATCH = 10
        WORKER_POLL_BLOCK_MS = 10
        RAG_STREAM = "rag:ingest"
        RAG_CONSUMER_GROUP = "ami-rag"
        PARSER = "mineru"
        PARSE_METHOD = "auto"
        CRAWL_IMAGES_ENABLED = False

    return IngestWorker(
        rag_anything=rag_anything,
        docs_repo=fake_docs_repo,
        state_repo=fake_state_repo,
        queue=fake_queue,
        asset_store=asset_store,
        settings=_S(),
    )


def _calls(rag_anything, kind):
    return [c for c in rag_anything.calls if c[0] == kind]


async def test_minio_parse_ingests_assets_and_counts(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    event = RagEvent(event="created", document_id=PDF_ID, content_hash="hash-a")
    result = await worker.handle_event(event)

    assert fake_asset_store.fetched == ["documents/HV/a.pdf"]
    parses = _calls(fake_rag_anything, "parse")
    assert len(parses) == 1 and parses[0][2] == "auto"

    inserts = _calls(fake_rag_anything, "insert")
    assert len(inserts) == 1
    _, content_list, file_path, doc_id = inserts[0]
    assert doc_id == PDF_ID
    assert file_path == f"{PDF_ID}_a.pdf"
    image = next(i for i in content_list if i["type"] == "image")
    assert image["asset_key"] == f"rag-assets/{PDF_ID}/img1.png"

    assert fake_asset_store.load_content_list(PDF_ID) is not None
    assert result["source"] == "minio_parse"
    assert result["assets"] == [f"rag-assets/{PDF_ID}/img1.png"]
    assert result["counts"] == {"text": 1, "image": 1, "table": 1, "equation": 0}
    assert result["page_count"] == 3
    assert result["source_hash"] == source_hash(fake_docs_repo.docs[PDF_ID])


async def test_docx_is_minio_parse(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    result = await worker.handle_event(RagEvent(event="created", document_id=DOCX_ID))

    assert result["source"] == "minio_parse"
    assert fake_asset_store.fetched == ["documents/HV/d.docx"]


async def test_mongo_text_ingests_content(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    result = await worker.handle_event(RagEvent(event="created", document_id=TEXT_ID))

    assert _calls(fake_rag_anything, "parse") == []
    assert fake_asset_store.fetched == []
    _, content_list, file_path, doc_id = _calls(fake_rag_anything, "insert")[0]
    assert content_list == [{"type": "text", "text": "nội dung text E", "page_idx": 0}]
    assert doc_id == TEXT_ID
    assert file_path == f"{TEXT_ID}_text"
    assert result["source"] == "mongo_text"
    assert result["counts"] == {"text": 1, "image": 0, "table": 0, "equation": 0}
    assert result["page_count"] == 1
    assert result["assets"] == []
    assert fake_asset_store.load_content_list(TEXT_ID) is not None


async def test_txt_upload_uses_mongo_text(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    result = await worker.handle_event(RagEvent(event="created", document_id=TXT_ID))

    assert result["source"] == "mongo_text"
    assert _calls(fake_rag_anything, "parse") == []


async def test_crawl_document_uses_source_url_as_file_path(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    await worker.handle_event(RagEvent(event="created", document_id=CRAWL_ID))

    _, _, file_path, _ = _calls(fake_rag_anything, "insert")[0]
    assert file_path == f"{CRAWL_ID}_b"


async def test_unchanged_source_skipped_and_state_not_overwritten(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    h = source_hash(fake_docs_repo.docs[PDF_ID])
    fake_state_repo.mark_processed(
        PDF_ID, h, source="minio_parse", counts={"text": 9}, assets=["k"]
    )

    assert await worker.handle_event(RagEvent(event="updated", document_id=PDF_ID)) is None
    assert fake_rag_anything.calls == []

    await worker._process_message("1-1", RagEvent(event="updated", document_id=PDF_ID))
    assert "1-1" in fake_queue.acked
    assert fake_state_repo.state[PDF_ID]["counts"] == {"text": 9}
    assert fake_state_repo.state[PDF_ID]["assets"] == ["k"]


async def test_updated_event_delete_reinsert(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    fake_state_repo.mark_processed(PDF_ID, "hash-old")
    await worker.handle_event(RagEvent(event="updated", document_id=PDF_ID))

    assert ("delete", PDF_ID) in fake_rag_anything.calls
    assert PDF_ID in fake_asset_store.deleted_assets
    assert len(_calls(fake_rag_anything, "insert")) == 1


async def test_stale_row_is_reingested(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    fake_state_repo.mark_processed(TEXT_ID, source_hash(fake_docs_repo.docs[TEXT_ID]))
    fake_state_repo.mark_stale([TEXT_ID])
    await worker.handle_event(RagEvent(event="updated", document_id=TEXT_ID))

    assert len(_calls(fake_rag_anything, "insert")) == 1


async def test_deleted_event_purges_index_assets_and_row(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    fake_state_repo.mark_processed(PDF_ID, "hash-a")
    await worker._process_message("1-1", RagEvent(event="deleted", document_id=PDF_ID))

    assert ("delete", PDF_ID) in fake_rag_anything.calls
    assert PDF_ID in fake_asset_store.deleted_assets
    assert fake_state_repo.state.get(PDF_ID) is None
    assert "1-1" in fake_queue.acked


async def test_missing_document_skipped(
    fake_rag_anything, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, FakeDocsRepo(), fake_state_repo, fake_queue, fake_asset_store
    )
    result = await worker.handle_event(
        RagEvent(event="created", document_id="64b000000000000000000009")
    )
    assert result is None
    assert fake_rag_anything.calls == []


async def test_empty_text_content_skipped(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    fake_docs_repo.docs[TEXT_ID]["content"] = "   "
    assert await worker.handle_event(RagEvent(event="created", document_id=TEXT_ID)) is None
    assert fake_rag_anything.calls == []


async def test_pdf_without_file_path_fails_after_max_delivery(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    fake_docs_repo.docs[PDF_ID]["file_path"] = None
    worker = _worker(
        fake_rag_anything,
        fake_docs_repo,
        fake_state_repo,
        fake_queue,
        fake_asset_store,
        max_delivery=2,
    )
    event = RagEvent(event="created", document_id=PDF_ID)
    await worker._process_message("1-1", event)
    assert "1-1" not in fake_queue.acked
    await worker._process_message("1-1", event)

    assert fake_state_repo.state[PDF_ID]["status"] == "failed"
    assert "file_path" in fake_state_repo.state[PDF_ID]["error"]
    assert "1-1" in fake_queue.acked
    assert fake_rag_anything.calls == []


async def test_failure_increments_attempts_and_keeps_pending(
    fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        FakeRAGAnything(fail_insert=True),
        fake_docs_repo,
        fake_state_repo,
        fake_queue,
        fake_asset_store,
    )
    await worker._process_message("1-1", RagEvent(event="created", document_id=PDF_ID))

    assert fake_state_repo.state[PDF_ID]["attempts"] == 1
    assert fake_state_repo.state[PDF_ID]["status"] != "failed"
    assert "1-1" not in fake_queue.acked


async def test_failure_after_max_delivery_marks_failed_and_acks(
    fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        FakeRAGAnything(fail_insert=True),
        fake_docs_repo,
        fake_state_repo,
        fake_queue,
        fake_asset_store,
        max_delivery=3,
    )
    event = RagEvent(event="created", document_id=PDF_ID)
    for _ in range(3):
        await worker._process_message("1-1", event)

    assert fake_state_repo.state[PDF_ID]["status"] == "failed"
    assert "1-1" in fake_queue.acked


async def test_success_marks_processed_and_acks(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    await worker._process_message("1-1", RagEvent(event="created", document_id=PDF_ID))

    row = fake_state_repo.state[PDF_ID]
    assert "1-1" in fake_queue.acked
    assert row["status"] == "processed"
    assert row["source_hash"] == source_hash(fake_docs_repo.docs[PDF_ID])
    assert row["source"] == "minio_parse"
    assert row["parser"] == "mineru"
    assert row["counts"]["table"] == 1
    assert row["page_count"] == 3
    assert row["assets"] == [f"rag-assets/{PDF_ID}/img1.png"]


async def test_skipped_event_releases_attempt_and_leaves_no_orphan_row(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )

    # unknown document: begin_attempt created a row, the skip must drop it again
    missing = "64b000000000000000000009"
    await worker._process_message("1-1", RagEvent(event="created", document_id=missing))
    assert missing not in fake_state_repo.state
    assert "1-1" in fake_queue.acked

    # unchanged document: processed row keeps attempts at 0 across repeated skips
    await worker._process_message("1-2", RagEvent(event="created", document_id=TEXT_ID))
    for i in range(3):
        await worker._process_message(f"2-{i}", RagEvent(event="updated", document_id=TEXT_ID))
    assert fake_state_repo.state[TEXT_ID]["status"] == "processed"
    assert fake_state_repo.state[TEXT_ID]["attempts"] == 0


async def test_parse_cache_entry_dropped_after_parse(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    await worker.handle_event(RagEvent(event="created", document_id=PDF_ID))
    assert len(fake_rag_anything.parse_cache.deleted) == 1
    assert fake_rag_anything.parse_cache.deleted[0].startswith("cache::")


async def test_parse_cache_cleanup_failure_does_not_fail_ingest(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    fake_rag_anything.parse_cache = None  # cleanup raises AttributeError internally
    result = await worker.handle_event(RagEvent(event="created", document_id=PDF_ID))
    assert result is not None and result["source"] == "minio_parse"


async def test_failed_document_gets_fresh_attempt_budget_for_reprocess(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    fake_docs_repo.docs[PDF_ID]["file_path"] = None
    worker = _worker(
        fake_rag_anything,
        fake_docs_repo,
        fake_state_repo,
        fake_queue,
        fake_asset_store,
        max_delivery=1,
    )
    await worker._process_message("1-1", RagEvent(event="created", document_id=PDF_ID))
    assert fake_state_repo.state[PDF_ID]["status"] == "failed"
    assert fake_state_repo.state[PDF_ID]["attempts"] == 0


async def test_registry_row_gets_link_fields_from_document(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    from bson import ObjectId

    fake_docs_repo.docs[PDF_ID]["owner_id"] = "sub-1"
    worker = _worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    await worker._process_message("1-1", RagEvent(event="created", document_id=PDF_ID))

    row = fake_state_repo.state[PDF_ID]
    assert row["status"] == "processed"
    assert row["document_oid"] == ObjectId(PDF_ID)
    assert row["organization_unit_id"] == fake_docs_repo.docs[PDF_ID]["organization_unit_id"]
    assert row["owner_id"] == "sub-1"
    assert row["document_type"] == fake_docs_repo.docs[PDF_ID]["document_type"]
