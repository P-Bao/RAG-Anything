"""Fusion of per-modality rerank result lists into a single ranked page.

The query path splits retrieval into two branches (see
`ami_rag.api.routes.rag`), each reranked in a single call:

- ``text+table``: prose and tables share one pool and one text-only rerank pass.
  Tables are serialized to text for the reranker, which is cheap (~13 ms/doc)
  and puts both on the same score scale.
- ``image``: reranked with the VL reranker, because image scores sit on their
  own scale and that gap has no equivalent between tables and prose. Keeping
  images in their own branch is what makes one calibration fix enough.

Before merging, an image gate drops images that cannot reach the page, so no
image payload is carried into the answer for a chunk that will not be used.

Merging several independently ranked lists back into one `top_k` page happens
here. Fusion is pure Python with no numpy or scikit-learn dependency, matching
`ami_rag.core.calibration`, to keep the container image small.

Supported strategies:

- ``calibrated``: global sort by calibrated probability P(relevant | modality).
  Score scales are otherwise incomparable across pools -- a raw rerank score of
  1.0 for text is weaker evidence than 0.14 for an image.
- ``rrf``: Reciprocal Rank Fusion, ``score = weight / (k + rank)``. Scale free.
  With disjoint pools and equal weights this degenerates to a round-robin
  interleave by rank, so ``k`` does not change the output; ``k`` only matters
  for overlapping pools or per-pool weights.
- ``quota``: reserve slots per pool (best-by-rank inside each pool, above an
  optional score floor), then fill the remaining slots globally by calibrated
  probability. Reserved slots are returned to the global fill when a pool is
  empty or has nothing above its floor.

Every strategy consumes ``dict[pool_name] -> list[(chunk, raw_score)]`` already
sorted by that pool's reranker, and returns `list[FusedHit]`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

# Pool holding chunks whose modality is in no pool of RETRIEVAL_FUSION_POOLS
# (equation, generic, or a missing modality field).
OTHER_POOL = "other"

# Modality that needs the VL reranker. Only a pool containing it is reranked
# with images; every other pool is reranked as plain text.
IMAGE_MODALITY = "image"

STRATEGIES = ("calibrated", "rrf", "quota")


@dataclass(frozen=True)
class QuotaRule:
    """Reserved slots for one pool, with an optional calibrated-score floor."""

    slots: int
    min_score: float = 0.0


@dataclass(frozen=True)
class FusedHit:
    """One fused result: the chunk, the fused ranking score and its origin."""

    chunk: dict
    score: float
    raw_score: float | None
    pool: str
    rank: int


def _parse_assignment(token: str, kind: str) -> tuple[str, str]:
    name, sep, value = token.partition("=")
    if not sep:
        raise ValueError(f"Fusion {kind} spec entry must be 'name=value', got {token!r}")
    name = name.strip().lower()
    if not name:
        raise ValueError(f"Fusion {kind} spec entry has an empty name: {token!r}")
    return name, value.strip()


def parse_pool_groups(spec: str) -> dict[str, tuple[str, ...]]:
    """Parse ``"text+table,image"`` into pool name -> the modalities it holds.

    ``+`` joins several modalities into one pool, so prose and tables can share
    a pool (and a single rerank pass) while images stay isolated. A pool name is
    its own first modality, which keeps single-modality pools written as
    ``image=15`` rather than ``image=image=15``.
    """
    groups: dict[str, tuple[str, ...]] = {}
    for token in (spec or "").split(","):
        if not token.strip():
            continue
        modalities: list[str] = []
        for raw in token.split("+"):
            name = raw.strip().lower()
            if not name:
                continue
            if name == OTHER_POOL:
                raise ValueError(
                    f"'{OTHER_POOL}' is reserved for unlisted modalities, "
                    "remove it from the pool spec"
                )
            if name not in modalities:
                modalities.append(name)
        if not modalities:
            continue
        name = modalities[0]
        if name in groups:
            raise ValueError(f"Fusion pool {name!r} is declared twice in {spec!r}")
        groups[name] = tuple(modalities)
    if not groups:
        raise ValueError(f"Fusion pool spec is empty: {spec!r}")
    return groups


def _reject_modality_group(name: str, kind: str) -> None:
    """Sizes and quotas key on a pool *name*, never on a joined modality group.

    Without this the spec would parse cleanly and then match nothing, so a quota
    for a `text+table` pool would silently reserve nothing at all.
    """
    if "+" in name:
        head = name.split("+")[0]
        raise ValueError(
            f"Fusion {kind} must be keyed by pool name, not modality group: "
            f"{name!r} -> {head!r}"
        )


def parse_pool_sizes(spec: str) -> dict[str, int]:
    """Parse ``"text+table=40,image=15"`` into per-pool fetch depth."""
    sizes: dict[str, int] = {}
    for token in (spec or "").split(","):
        if not token.strip():
            continue
        name, raw = _parse_assignment(token, "pool size")
        _reject_modality_group(name, "pool size")
        try:
            size = int(raw)
        except ValueError:
            raise ValueError(
                f"Fusion pool size for {name!r} must be an integer, got {raw!r}"
            ) from None
        if size < 0:
            raise ValueError(f"Fusion pool size for {name!r} must be >= 0, got {size}")
        sizes[name] = size
    if not sizes:
        raise ValueError(f"Fusion pool sizes spec is empty: {spec!r}")
    return sizes


def parse_quota(spec: str) -> dict[str, QuotaRule]:
    """Parse ``"table=2@0.45,image=2@0.40"`` into reserved slots per pool.

    The ``@threshold`` suffix is a floor on the calibrated score; a pool only
    consumes a reserved slot for a candidate scoring at or above it.
    """
    quotas: dict[str, QuotaRule] = {}
    for token in (spec or "").split(","):
        if not token.strip():
            continue
        name, raw = _parse_assignment(token, "quota")
        _reject_modality_group(name, "quota")
        slots_text, at, floor_text = raw.partition("@")
        try:
            slots = int(slots_text)
        except ValueError:
            raise ValueError(
                f"Fusion quota slots for {name!r} must be an integer, got {slots_text!r}"
            ) from None
        if slots < 0:
            raise ValueError(f"Fusion quota slots for {name!r} must be >= 0, got {slots}")
        min_score = 0.0
        if at:
            try:
                min_score = float(floor_text)
            except ValueError:
                raise ValueError(
                    f"Fusion quota floor for {name!r} must be a number, got {floor_text!r}"
                ) from None
        quotas[name] = QuotaRule(slots=slots, min_score=min_score)
    return quotas


def resolve_pool(modality: str | None, pool_groups: Mapping[str, Sequence[str]]) -> str:
    """Map a chunk modality to its pool name (unlisted modalities go to ``other``).

    Several modalities can resolve to the same pool, which is what lets prose
    and tables share one rerank pass.
    """
    name = (modality or "").strip().lower()
    for pool_name, modalities in pool_groups.items():
        if name in modalities:
            return pool_name
    return OTHER_POOL


def pool_is_multimodal(pool_name: str, pool_groups: Mapping[str, Sequence[str]]) -> bool:
    """Whether a pool must be reranked with images rather than as plain text.

    Only the image pool needs the VL reranker; reranking a table as plain text
    is both cheaper and score-comparable with prose.
    """
    return IMAGE_MODALITY in pool_groups.get(pool_name, ())


def default_score_of(chunk: dict, raw_score: float | None) -> float:
    """Ranking score used when no calibrator is configured: the raw rerank score."""
    return float(raw_score) if raw_score is not None else 0.0


def rrf_score(rank: int, k: int = 60, weight: float = 1.0) -> float:
    """Reciprocal rank score, normalized so that rank 1 with weight 1 scores 1.0."""
    if k < 1:
        raise ValueError(f"RRF k must be >= 1, got {k}")
    if rank < 1:
        raise ValueError(f"RRF rank must be >= 1, got {rank}")
    return weight * (k + 1) / (k + rank)


def _identity(chunk: dict) -> str:
    """Stable identity for cross-pool de-duplication."""
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return "|".join(
        str(chunk.get(field) or "")[:80]
        for field in ("doc_id", "modality", "asset_key", "content")
    )


def _ranked_pools(
    pools: Mapping[str, Iterable[tuple[dict, float | None]]],
) -> list[tuple[str, list[tuple[dict, float | None]]]]:
    """Materialize pools, dropping items already emitted by an earlier pool."""
    ordered: list[tuple[str, list[tuple[dict, float | None]]]] = []
    seen: set[str] = set()
    for pool_name, items in pools.items():
        kept: list[tuple[dict, float | None]] = []
        for chunk, raw_score in items or []:
            key = _identity(chunk)
            if key in seen:
                continue
            seen.add(key)
            kept.append((chunk, raw_score))
        ordered.append((pool_name, kept))
    return ordered


def _emit(
    chunk: dict,
    raw_score: float | None,
    score: float,
    pool: str,
    rank: int,
) -> FusedHit:
    return FusedHit(chunk=chunk, score=score, raw_score=raw_score, pool=pool, rank=rank)


def gate_image_hits(
    reranked: Mapping[str, Iterable[tuple[dict, float | None]]],
    *,
    pool_groups: Mapping[str, Sequence[str]],
    top_k: int,
    score_of: Callable[[dict, float | None], float] = default_score_of,
) -> tuple[dict[str, list[tuple[dict, float | None]]], int]:
    """Drop images that cannot reach the page, before any merging happens.

    An image survives when its score reaches the score of the last text chunk
    that would still make the page -- the cut line. That is exactly the evidence
    an image needs to displace prose, and it is deliberately not the best text
    score: a weak image that beats the runner-up prose is still worth answering
    with, it just must not outrank strong prose on its own.

    Under ``calibrated`` this filter cannot change which documents reach the
    page, because a chunk below the cut line cannot win a global sort either.
    What it buys is bounding the work: every dropped image is an asset fetch and
    an image block in the answer, and those are paid for even when the chunk
    loses. It does change ``quota``, where a filtered image frees its reserved
    slot to another pool.
    """
    pools = {name: list(items or []) for name, items in reranked.items()}
    text_pools = [name for name in pools if not pool_is_multimodal(name, pool_groups)]
    image_pools = [name for name in pools if pool_is_multimodal(name, pool_groups)]
    if not image_pools:
        return pools, 0

    text_scores = sorted(
        (score_of(chunk, raw) for name in text_pools for chunk, raw in pools[name]),
        reverse=True,
    )
    if not text_scores:
        # Nothing to compare against: keep every image rather than starve the
        # branch that exists precisely because images score on a different scale.
        return pools, 0
    cut = text_scores[min(top_k, len(text_scores)) - 1]

    dropped = 0
    for name in image_pools:
        kept = []
        for chunk, raw in pools[name]:
            # An unmeasured chunk is not a weak chunk: a pool whose rerank failed
            # yields unscored candidates, and dropping the whole branch here would
            # turn a transient error into a silently missing modality. Let the
            # merge decide those instead.
            if raw is None or score_of(chunk, raw) >= cut:
                kept.append((chunk, raw))
            else:
                dropped += 1
        pools[name] = kept
    return pools, dropped


def fuse_calibrated(
    pools: Mapping[str, Iterable[tuple[dict, float | None]]],
    *,
    top_k: int,
    score_of: Callable[[dict, float | None], float] = default_score_of,
) -> list[FusedHit]:
    """Sort every pool's candidates together by calibrated probability."""
    hits = [
        _emit(chunk, raw, float(score_of(chunk, raw)), pool, rank)
        for pool, items in _ranked_pools(pools)
        for rank, (chunk, raw) in enumerate(items, start=1)
    ]
    hits.sort(key=lambda hit: (-hit.score, hit.pool, hit.rank))
    return hits[:top_k]


def fuse_rrf(
    pools: Mapping[str, Iterable[tuple[dict, float | None]]],
    *,
    top_k: int,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[FusedHit]:
    """Reciprocal Rank Fusion; ``score = weight * (k + 1) / (k + rank)``."""
    hits = [
        _emit(chunk, raw, rrf_score(rank, k, (weights or {}).get(pool, 1.0)), pool, rank)
        for pool, items in _ranked_pools(pools)
        for rank, (chunk, raw) in enumerate(items, start=1)
    ]
    hits.sort(key=lambda hit: (-hit.score, hit.pool, hit.rank))
    return hits[:top_k]


def fuse_quota(
    pools: Mapping[str, Iterable[tuple[dict, float | None]]],
    *,
    top_k: int,
    quotas: Mapping[str, QuotaRule],
    score_of: Callable[[dict, float | None], float] = default_score_of,
) -> list[FusedHit]:
    """Reserve slots per pool (best by rank, above a score floor), then fill by score.

    A reserved slot is only consumed by a candidate scoring at or above the
    pool's ``min_score``; otherwise the slot is released back to the global fill.

    Quota decides *membership* of the page, score decides *order*. Reserved
    candidates are kept unconditionally and only the leftover slots are filled
    globally, then the whole page is sorted by score -- so a low-scoring but
    reserved candidate cannot be pushed out by many stronger leftovers, while
    ``score`` still decreases monotonically down the page.
    """
    ranked = _ranked_pools(pools)
    reserved: list[FusedHit] = []
    # Identifies a candidate by (pool, position) -- positions repeat across pools.
    claimed: set[tuple[str, int]] = set()

    for pool_name, items in ranked:
        rule = quotas.get(pool_name)
        if rule is None:
            continue
        # A pool can only spend as many reserved slots as it has candidates.
        for index, (chunk, raw) in enumerate(items[: rule.slots]):
            score = float(score_of(chunk, raw))
            if score < rule.min_score:
                # Pools are rank-ordered, so this one is below the floor as well.
                break
            reserved.append(_emit(chunk, raw, score, pool_name, index + 1))
            claimed.add((pool_name, index))

    page = reserved[:top_k]
    remaining = top_k - len(page)

    leftovers = [
        _emit(chunk, raw, float(score_of(chunk, raw)), pool_name, index + 1)
        for pool_name, items in ranked
        for index, (chunk, raw) in enumerate(items)
        if (pool_name, index) not in claimed
    ]
    leftovers.sort(key=lambda hit: (-hit.score, hit.pool, hit.rank))

    page.extend(leftovers[:remaining])
    page.sort(key=lambda hit: (-hit.score, hit.pool, hit.rank))
    return page


def fuse(
    pools: Mapping[str, Iterable[tuple[dict, float | None]]],
    *,
    strategy: str,
    top_k: int,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    quotas: Mapping[str, QuotaRule] | None = None,
    score_of: Callable[[dict, float | None], float] = default_score_of,
) -> list[FusedHit]:
    """Dispatch to the configured fusion strategy.

    ``strategy`` is one of `STRATEGIES`; ``"quota"`` falls back to calibrated
    ordering when no quota is configured, and ``"rrf"`` ignores ``score_of``
    because it ranks by position rather than by score.
    """
    if top_k < 0:
        raise ValueError(f"Fusion top_k must be >= 0, got {top_k}")
    if strategy == "calibrated":
        return fuse_calibrated(pools, top_k=top_k, score_of=score_of)
    if strategy == "rrf":
        return fuse_rrf(pools, top_k=top_k, k=k, weights=weights)
    if strategy == "quota":
        if not quotas:
            return fuse_calibrated(pools, top_k=top_k, score_of=score_of)
        return fuse_quota(pools, top_k=top_k, quotas=quotas, score_of=score_of)
    raise ValueError(f"Unknown fusion strategy {strategy!r} (expected one of {', '.join(STRATEGIES)})")