"""Offline tests for per-modality result fusion (no network, no services)."""

import pytest

from ami_rag.core.fusion import (
    OTHER_POOL,
    FusedHit,
    QuotaRule,
    default_score_of,
    fuse,
    fuse_calibrated,
    fuse_quota,
    fuse_rrf,
    gate_image_hits,
    parse_pool_groups,
    parse_pool_sizes,
    parse_quota,
    pool_is_multimodal,
    resolve_pool,
    rrf_score,
)


def _pool(prefix: str, modality: str, count: int, scores=None):
    """Build one ranked pool of ``count`` chunks with descending raw scores."""
    items = []
    for i in range(1, count + 1):
        raw = (scores[i - 1] if scores else 1.0 / i) if scores else None
        chunk = {
            "chunk_id": f"{prefix}-{i}",
            "modality": modality,
            "content": f"{prefix} content {i}",
        }
        items.append((chunk, raw))
    return items


# --- spec parsing -----------------------------------------------------------


def test_parse_pool_specs():
    assert parse_pool_groups("text+table,image") == {
        "text": ("text", "table"),
        "image": ("image",),
    }
    assert parse_pool_groups(" image , TEXT+Table ") == {
        "image": ("image",),
        "text": ("text", "table"),
    }
    # A pool name is its own first modality, so "+" is optional and idempotent.
    assert parse_pool_groups("image+image") == {"image": ("image",)}
    # Sizes are keyed by pool name, which is the pool's first modality.
    assert parse_pool_sizes("text=40,image=15") == {
        "text": 40,
        "image": 15,
    }
    assert parse_quota("") == {}
    quotas = parse_quota("text=3@0.45, image=2 ")
    assert quotas["text"] == QuotaRule(slots=3, min_score=0.45)
    assert quotas["image"] == QuotaRule(slots=2, min_score=0.0)


def test_parse_pool_groups_rejects_duplicate_and_reserved():
    with pytest.raises(ValueError):
        parse_pool_groups("text+table,image,text")
    with pytest.raises(ValueError):
        parse_pool_groups("text+other")
    with pytest.raises(ValueError):
        parse_pool_sizes("text+table=40")
    with pytest.raises(ValueError):
        parse_quota("text+table=3")


@pytest.mark.parametrize(
    "parser, spec",
    [
        (parse_pool_groups, ""),
        (parse_pool_groups, "other"),
        (parse_pool_groups, "+"),
        (parse_pool_sizes, ""),
        (parse_pool_sizes, "text"),
        (parse_pool_sizes, "text=many"),
        (parse_pool_sizes, "text=-1"),
        (parse_quota, "table"),
        (parse_quota, "table=x"),
        (parse_quota, "table=-1"),
        (parse_quota, "table=2@nope"),
    ],
)
def test_parse_errors_are_explicit(parser, spec):
    with pytest.raises(ValueError):
        parser(spec)


def test_resolve_pool_maps_several_modalities_to_one_pool():
    groups = parse_pool_groups("text+table,image")
    assert resolve_pool("text", groups) == "text"
    assert resolve_pool("table", groups) == "text"   # shares the text rerank pass
    assert resolve_pool("IMAGE", groups) == "image"
    assert resolve_pool("equation", groups) == OTHER_POOL
    assert resolve_pool(None, groups) == OTHER_POOL


def test_only_the_image_pool_is_reranked_with_images():
    groups = parse_pool_groups("text+table,image")
    assert pool_is_multimodal("image", groups) is True
    assert pool_is_multimodal("text", groups) is False


# --- RRF --------------------------------------------------------------------


def test_rrf_score_formula():
    assert rrf_score(1, k=60) == pytest.approx(1.0)
    assert rrf_score(2, k=60) == pytest.approx(61 / 62)
    assert rrf_score(1, k=60, weight=2.0) == pytest.approx(2.0)


def test_rrf_ignores_score_magnitude():
    """RRF is scale free: a pool with terrible raw scores still ranks by position."""
    strong = fuse_rrf({"text": _pool("t", "text", 3, [0.99, 0.5, 0.1])}, top_k=3)
    weak = fuse_rrf({"text": _pool("t", "text", 3, [0.01, 0.001, 0.0001])}, top_k=3)
    assert [h.score for h in strong] == [h.score for h in weak]
    assert [h.chunk["chunk_id"] for h in strong] == ["t-1", "t-2", "t-3"]


@pytest.mark.parametrize("k", [1, 10, 20, 60, 200])
def test_rrf_k_is_irrelevant_for_disjoint_pools(k):
    """Documented consequence of disjoint pools + equal weights: ordering is rank only."""
    pools = {
        "text": _pool("t", "text", 4),
        "image": _pool("i", "image", 4),
    }
    baseline = [h.chunk["chunk_id"] for h in fuse_rrf(pools, top_k=8, k=60)]
    assert [h.chunk["chunk_id"] for h in fuse_rrf(pools, top_k=8, k=k)] == baseline
    # Round-robin interleave by pool rank, independent of k. Ties are broken by
    # pool name then rank, so "image" comes before "text".
    assert baseline == ["i-1", "t-1", "i-2", "t-2", "i-3", "t-3", "i-4", "t-4"]


def test_rrf_weights_break_the_k_invariance():
    pools = {"text": _pool("t", "text", 3), "image": _pool("i", "image", 3)}
    weighted = fuse_rrf(pools, top_k=3, k=60, weights={"image": 5.0})
    assert [h.pool for h in weighted] == ["image", "image", "image"]


# --- calibrated -------------------------------------------------------------


def test_calibrated_sorts_across_pools_by_score():
    pools = {
        "text": _pool("t", "text", 2, [0.99, 0.98]),
        "image": _pool("i", "image", 2, [0.90, 0.10]),
    }
    hits = fuse_calibrated(pools, top_k=4)
    assert [h.chunk["chunk_id"] for h in hits] == ["t-1", "t-2", "i-1", "i-2"]
    assert [h.score for h in hits] == [0.99, 0.98, 0.90, 0.10]
    assert hits[0].raw_score == 0.99
    assert hits[0].pool == "text"


def test_score_of_receives_chunk_modality():
    """Calibrator-style callable: score depends on the chunk, not on the pool."""
    seen = []

    def score_of(chunk, raw):
        seen.append((chunk["modality"], raw))
        return raw * (10.0 if chunk["modality"] == "image" else 1.0)

    hits = fuse_calibrated(
        {"text": _pool("t", "text", 1, [0.5]), "image": _pool("i", "image", 1, [0.2])},
        top_k=2,
        score_of=score_of,
    )
    assert [h.chunk["chunk_id"] for h in hits] == ["i-1", "t-1"]
    assert sorted(seen) == [("image", 0.2), ("text", 0.5)]


# --- quota ------------------------------------------------------------------


def test_quota_reserves_slots_and_fills_by_score():
    # image scores look tiny next to text, but each pool contributes its best.
    pools = {
        "text": _pool("t", "text", 6, [0.99, 0.98, 0.97, 0.96, 0.95, 0.94]),
        "image": _pool("i", "image", 3, [0.14, 0.03, 0.02]),
        "table": _pool("b", "table", 2, [0.07, 0.01]),
    }
    hits = fuse_quota(
        pools, top_k=5, quotas={"table": QuotaRule(2), "image": QuotaRule(1)}
    )
    ids = [h.chunk["chunk_id"] for h in hits]
    # 3 reserved slots survive even though their raw scores are far below text;
    # only the 2 leftover slots go to the strongest text candidates.
    assert ids == ["t-1", "t-2", "i-1", "b-1", "b-2"]
    # Membership respects the quota, order still follows score descending.
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len(ids) == 5


def test_quota_floor_releases_slot_back_to_global_fill():
    pools = {
        "text": _pool("t", "text", 2, [0.99, 0.98]),
        "image": _pool("i", "image", 2, [0.01, 0.005]),
    }
    strict = fuse_quota(
        pools, top_k=3, quotas={"image": QuotaRule(1, min_score=0.5)}
    )
    assert [h.chunk["chunk_id"] for h in strict] == ["t-1", "t-2", "i-1"]

    loose = fuse_quota(pools, top_k=3, quotas={"image": QuotaRule(1)})
    assert [h.chunk["chunk_id"] for h in loose] == ["t-1", "t-2", "i-1"]


def test_quota_with_empty_pool_does_not_starve_the_page():
    pools = {"text": _pool("t", "text", 3, [0.9, 0.8, 0.7])}
    hits = fuse_quota(pools, top_k=3, quotas={"image": QuotaRule(2, min_score=0.5)})
    assert [h.chunk["chunk_id"] for h in hits] == ["t-1", "t-2", "t-3"]


def test_quota_slots_exceed_top_k_keep_best_reserved():
    pools = {
        "text": _pool("t", "text", 1, [0.1]),
        "image": _pool("i", "image", 5, [0.9, 0.8, 0.7, 0.6, 0.5]),
    }
    hits = fuse_quota(pools, top_k=2, quotas={"image": QuotaRule(5)})
    assert [h.chunk["chunk_id"] for h in hits] == ["i-1", "i-2"]


def test_quota_indices_do_not_leak_between_pools():
    """Two pools both holding a candidate at position 1 must each keep their own."""
    pools = {
        "text": _pool("t", "text", 2, [0.9, 0.8]),
        "table": _pool("b", "table", 2, [0.7, 0.6]),
    }
    hits = fuse_quota(pools, top_k=4, quotas={"table": QuotaRule(1)})
    assert {h.chunk["chunk_id"] for h in hits} == {"t-1", "t-2", "b-1", "b-2"}


# --- shared behaviour -------------------------------------------------------


def test_cross_pool_duplicates_kept_once():
    shared = {"chunk_id": "shared", "modality": "text", "content": "same"}
    pools = {
        "text": [(shared, 0.9), ({"chunk_id": "t-2", "modality": "text"}, 0.5)],
        "other": [(dict(shared), 0.8)],
    }
    hits = fuse_calibrated(pools, top_k=5)
    assert [h.chunk["chunk_id"] for h in hits] == ["shared", "t-2"]


def test_missing_raw_score_scores_zero():
    pools = {
        "text": [({"chunk_id": "t-1", "modality": "text"}, None)],
        "image": [({"chunk_id": "i-1", "modality": "image"}, 0.01)],
    }
    hits = fuse_calibrated(pools, top_k=2)
    assert [h.chunk["chunk_id"] for h in hits] == ["i-1", "t-1"]
    assert hits[1].raw_score is None
    assert default_score_of({}, None) == 0.0


def test_fuse_dispatch_and_validation():
    pools = {"text": _pool("t", "text", 2), "image": _pool("i", "image", 2)}

    for strategy, expected in [
        ("calibrated", fuse_calibrated(pools, top_k=3)),
        ("rrf", fuse_rrf(pools, top_k=3, k=60)),
        # quota without a configured quota behaves like calibrated.
        ("quota", fuse_calibrated(pools, top_k=3)),
    ]:
        dispatched = fuse(pools, strategy=strategy, top_k=3)
        assert [h.chunk for h in dispatched] == [h.chunk for h in expected]
        assert [h.score for h in dispatched] == [h.score for h in expected]

    with_quotas = fuse(
        pools, strategy="quota", top_k=2, quotas={"image": QuotaRule(1)}
    )
    assert [h.chunk["chunk_id"] for h in with_quotas] == ["i-1", "i-2"]

    with pytest.raises(ValueError, match="Unknown fusion strategy"):
        fuse(pools, strategy="bogus", top_k=2)
    with pytest.raises(ValueError):
        fuse(pools, strategy="calibrated", top_k=-1)


def test_fused_hit_is_immutable():
    hit = FusedHit(chunk={}, score=0.1, raw_score=0.2, pool="text", rank=1)
    with pytest.raises(AttributeError):
        hit.score = 0.5  # type: ignore[misc]

# --- image gate -------------------------------------------------------------


def _gated(pools, top_k, score_of=None, groups="text+table,image"):
    return gate_image_hits(
        pools,
        pool_groups=parse_pool_groups(groups),
        top_k=top_k,
        score_of=score_of or default_score_of,
    )


def test_image_gate_keeps_images_at_or_above_the_text_cut_line():
    # Text cut line at top_k=2 is the 2nd best text chunk (0.50).
    pools = {
        "text": _pool("t", "text", 3, [0.9, 0.5, 0.1]),
        "image": _pool("i", "image", 3, [0.8, 0.5, 0.2]),
    }
    kept, dropped = _gated(pools, top_k=2)
    assert [c["chunk_id"] for c, _ in kept["text"]] == ["t-1", "t-2", "t-3"]
    assert [c["chunk_id"] for c, _ in kept["image"]] == ["i-1", "i-2"]
    assert dropped == 1
    # text is never gated.
    assert len(kept["text"]) == 3


def test_image_gate_is_not_a_threshold_against_the_best_text():
    # A mediocre image still beats the runner-up prose, so it survives.
    pools = {
        "text": _pool("t", "text", 3, [0.9, 0.5, 0.1]),
        "image": _pool("i", "image", 1, [0.6]),
    }
    kept, dropped = _gated(pools, top_k=2)
    assert [c["chunk_id"] for c, _ in kept["image"]] == ["i-1"]
    assert dropped == 0


def test_image_gate_uses_the_calibrated_score_not_the_raw_one():
    # Raw scores put image 0.6 below text 0.9, but after calibration the image is
    # the strongest chunk -- gating on raw scores would have thrown it away.
    def calibrated(chunk, raw):
        return 0.95 if chunk["modality"] == "image" else raw

    pools = {
        "text": _pool("t", "text", 3, [0.9, 0.5, 0.1]),
        "image": _pool("i", "image", 1, [0.6]),
    }
    kept, dropped = _gated(pools, top_k=2, score_of=calibrated)
    assert [c["chunk_id"] for c, _ in kept["image"]] == ["i-1"]
    assert dropped == 0


def test_image_gate_cut_line_uses_kth_text_not_kth_overall():
    pools = {
        "text": _pool("t", "text", 4, [0.9, 0.5, 0.4, 0.3]),
        "image": _pool("i", "image", 2, [0.45, 0.35]),
    }
    # top_k=3 -> cut line is the 3rd text score (0.4).
    kept, dropped = _gated(pools, top_k=3)
    assert [c["chunk_id"] for c, _ in kept["image"]] == ["i-1"]
    assert dropped == 1


def test_image_gate_keeps_everything_when_there_is_no_text_to_beat():
    pools = {"image": _pool("i", "image", 2, [0.02, 0.01])}
    kept, dropped = _gated(pools, top_k=5)
    assert [c["chunk_id"] for c, _ in kept["image"]] == ["i-1", "i-2"]
    assert dropped == 0


def test_image_gate_does_not_pad_the_page_with_images_below_all_prose():
    pools = {
        "text": _pool("t", "text", 2, [0.9, 0.5]),
        "image": _pool("i", "image", 2, [0.4, 0.3]),
    }
    # Fewer prose chunks than top_k, so the cut line is the worst prose (0.5).
    # top_k is a maximum, not a quota: the answer may come back short rather than
    # padded with an image that ranks below every chunk of prose.
    kept, dropped = _gated(pools, top_k=5)
    assert [c["chunk_id"] for c, _ in kept["image"]] == []
    assert dropped == 2


def test_image_gate_is_a_noop_for_pools_without_images():
    pools = {"text": _pool("t", "text", 3, [0.9, 0.5, 0.1])}
    kept, dropped = _gated(pools, top_k=2)
    assert dropped == 0
    assert [c["chunk_id"] for c, _ in kept["text"]] == ["t-1", "t-2", "t-3"]


def test_image_gate_frees_a_reserved_quota_slot_when_it_filters():
    pools = {
        "text": _pool("t", "text", 3, [0.9, 0.5, 0.1]),
        "image": _pool("i", "image", 2, [0.02, 0.01]),
    }
    gated, dropped = _gated(pools, top_k=2)
    assert dropped == 2
    hits = fuse_quota(gated, top_k=3, quotas={"image": QuotaRule(2)})
    assert [h.pool for h in hits] == ["text", "text", "text"]


def test_image_gate_keeps_unmeasured_images_for_the_merge_to_decide():
    """A failed rerank leaves raw_score=None; the gate must not eat the branch."""
    pools = {
        "text": _pool("t", "text", 3, [0.9, 0.5, 0.1]),
        "image": [
            ({"chunk_id": "i-1", "modality": "image", "content": "x"}, None),
        ],
    }
    kept, dropped = _gated(pools, top_k=2)
    assert [c["chunk_id"] for c, _ in kept["image"]] == ["i-1"]
    assert dropped == 0
