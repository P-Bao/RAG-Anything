#!/usr/bin/env python3
"""Sweep fusion rules offline against pool rankings collected from the live stack.

Reads `pool_rankings.json` (produced by `scripts/collect_pool_rankings.py`) and
evaluates `ami_rag.core.fusion` strategies without touching the embedder, Qdrant
or the rerank server, so a grid of quota slots / RRF k / pool depths / score
floors costs seconds instead of minutes.

Caveat: pool *depth* is simulated by truncating each pool's ranking to the first
N entries. A pool retrieved at depth N would be reranked as a smaller candidate
set, so its absolute scores shift slightly; treat depth numbers as an
approximation and confirm the winner with `scripts/run_fusion_bench.py`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ami_rag.core.fusion import (
    QuotaRule,
    fuse_calibrated,
    fuse_quota,
    fuse_rrf,
)

MODALITIES = ("text", "table", "image")


def _chunk(item: dict) -> dict:
    return {
        "chunk_id": item["chunk_id"],
        "doc_id": item["doc_id"],
        "modality": item["modality"],
    }


def _target(case: dict) -> tuple[str, str]:
    return ("doc_id", case["expected_doc_id"]) if case["modality"] == "text" else (
        "chunk_id",
        case["expected_chunk_id"],
    )


def _is_target(case: dict, hit_chunk: dict) -> bool:
    field, value = _target(case)
    return hit_chunk.get(field) == value


def _pools_for(case: dict, depths: dict[str, int], use_calibrated: bool) -> dict[str, list]:
    """Build the per-pool ranked lists, optionally truncated to `depths`."""
    pools: dict[str, list] = {}
    for pool, items in case["pools"].items():
        limit = depths.get(pool)
        sliced = items[:limit] if limit else items
        pools[pool] = [(_chunk(item), item["cal"] if use_calibrated else item["raw"]) for item in sliced]
    return pools


def _score_of(calibrated: bool):
    if calibrated:
        return lambda chunk, score: float(score if score is not None else 0.0)
    return lambda chunk, score: float(score if score is not None else 0.0)


def evaluate(data: dict, rule: dict) -> dict:
    """Run one fusion rule over every collected case."""
    depths = rule.get("depths") or {}
    strategy = rule["strategy"]
    use_cal = rule.get("use_calibrated", True)
    hits = {m: 0 for m in MODALITIES}
    totals = {m: 0 for m in MODALITIES}
    rr_sum = 0.0
    per_case = {}

    for case in data["cases"]:
        pools = _pools_for(case, depths, use_cal)
        if strategy == "calibrated":
            hits_list = fuse_calibrated(pools, top_k=5, score_of=_score_of(use_cal))
        elif strategy == "rrf":
            hits_list = fuse_rrf(pools, top_k=5, k=rule.get("k", 60))
        elif strategy == "quota":
            hits_list = fuse_quota(
                pools,
                top_k=5,
                quotas=rule["quotas"],
                score_of=_score_of(use_cal),
            )
        else:
            raise ValueError(f"unknown strategy {strategy}")

        modality = case["modality"]
        totals[modality] += 1
        rank = next(
            (i for i, hit in enumerate(hits_list, start=1) if _is_target(case, hit.chunk)),
            -1,
        )
        per_case[case["id"]] = {"rank": rank, "modality": modality, "xfail": case["xfail"]}
        if rank > 0:
            hits[modality] += 1
            rr_sum += 1.0 / rank

    total_hits = sum(hits.values())
    return {
        "hits": hits,
        "totals": totals,
        "hit_at_5": total_hits,
        "mrr": round(rr_sum / len(data["cases"]), 4),
        "per_case": per_case,
    }


def _parse_quota(spec: str) -> dict[str, QuotaRule]:
    rules = {}
    for token in spec.split(","):
        if not token.strip():
            continue
        name, _, rest = token.partition("=")
        slots, at, floor = rest.partition("@")
        rules[name.strip()] = QuotaRule(int(slots), float(floor) if at else 0.0)
    return rules


def _row(name: str, result: dict, baseline: int | None) -> str:
    hits = result["hits"]
    total = sum(result["totals"].values())
    mods = " ".join(f"{m[:3]}={hits[m]:>2}/{result['totals'][m]:<2}" for m in MODALITIES)
    delta = "" if baseline is None else f"{result['hit_at_5'] - baseline:+d}"
    return f"| {name:<34} | {result['hit_at_5']:>2}/{total:<3} | {mods:<26} | {result['mrr']:<7.4f} | {delta:>4} |"


def _gainers(per_case: dict, reference: dict) -> tuple[list[str], list[str]]:
    fixed = [cid for cid, c in reference.items() if c["rank"] < 0 and per_case[cid]["rank"] > 0]
    lost = [cid for cid, c in reference.items() if c["rank"] > 0 and per_case[cid]["rank"] < 0]
    return sorted(fixed), sorted(lost)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/tmp/opencode/pool_rankings.json")
    parser.add_argument("--baseline", type=int, default=19, help="measured Hit@5 of single mode")
    parser.add_argument("--grid", default="core", choices=["core", "depth", "all"])
    parser.add_argument("--detail", help="print per-case gains for this rule name")
    args = parser.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    print(f"Loaded {len(data['cases'])} cases, sizes={data['sizes']}, calibrator={data['has_calibrator']}")

    rules: list[tuple[str, dict]] = [
        ("calibrated (no quota)", {"strategy": "calibrated"}),
        ("calibrated, raw score (no calib)", {"strategy": "calibrated", "use_calibrated": False}),
        ("rrf k=60", {"strategy": "rrf", "k": 60}),
        ("rrf k=20", {"strategy": "rrf", "k": 20}),
        ("rrf k=10", {"strategy": "rrf", "k": 10}),
        ("quota t2i2 (no floor)", {"strategy": "quota", "quotas": _parse_quota("table=2,image=2")}),
        ("quota t2i2 floor .45/.40", {"strategy": "quota", "quotas": _parse_quota("table=2@0.45,image=2@0.40")}),
        ("quota t3i3 floor .50/.40", {"strategy": "quota", "quotas": _parse_quota("table=3@0.50,image=3@0.40")}),
        ("quota i5 floor .40", {"strategy": "quota", "quotas": _parse_quota("image=5@0.40")}),
        ("quota t3i5 floor .50/.30", {"strategy": "quota", "quotas": _parse_quota("table=3@0.50,image=5@0.30")}),
    ]
    if args.grid in ("depth", "all"):
        # NOTE: pool depth is NOT faithfully simulable offline. Truncating a pool
        # keeps the docs the reranker already ranked highest, whereas a shallower
        # *retrieval* would surface a different (vector-similarity) candidate set
        # -- e.g. table_08 sits at vector rank 43 but rerank rank 3, so it survives
        # a nominal depth of 10 here yet would never be retrieved at depth 10.
        # These rows are an upper bound on recall for a given depth; measure the
        # real trade-off with `scripts/run_fusion_bench.py`.
        for t, b, i, drop_other in itertools.product(
            [20], [10, 20, 50], [15, 30], [False, True]
        ):
            depths = {"text": t, "table": b, "image": i}
            if drop_other:
                depths["other"] = 0
            rules.append(
                (
                    f"calibrated depth t{t}/b{b}/i{i}{' no-other' if drop_other else ''}",
                    {"strategy": "calibrated", "depths": depths},
                )
            )
    if args.grid == "all":
        for t, b, i in itertools.product([20, 50], [20, 50], [15, 30]):
            rules.append(
                (
                    f"quota t3i3 .50/.30 depth {t}/{b}/{i}",
                    {
                        "strategy": "quota",
                        "quotas": _parse_quota("table=3@0.50,image=3@0.30"),
                        "depths": {"text": t, "table": b, "image": i},
                    },
                )
            )

    header = (
        f"| {'rule':<34} | {'HIT@5':<5} | {'per modality':<26} | {'MRR':<7} | {'vs base':>4} |"
    )
    print("\n" + "=" * 96)
    print("OFFLINE FUSION RULE SWEEP".center(96))
    print("=" * 96)
    print(header)
    print("-" * 96)

    results = {}
    for name, rule in rules:
        result = evaluate(data, rule)
        results[name] = result
        print(_row(name, result, args.baseline))
    print("-" * 96)
    print(f"baseline `single` (measured live): Hit@5 {args.baseline}/{len(data['cases'])}, MRR 0.6161")

    best_name, best = max(results.items(), key=lambda kv: (kv[1]["hit_at_5"], kv[1]["mrr"]))
    print(f"\nBest rule: {best_name} -> Hit@5 {best['hit_at_5']}/{len(data['cases'])}, MRR {best['mrr']}")

    rrf_equal = [results["rrf k=60"]["per_case"], results["rrf k=20"]["per_case"], results["rrf k=10"]["per_case"]]
    same = all(p == rrf_equal[0] for p in rrf_equal[1:])
    print(
        f"RRF k=60/20/10 identical output: {same} "
        "(expected: disjoint pools + equal weights => order depends on rank only)"
    )

    reference = {
        cid: c for cid, c in results["calibrated (no quota)"]["per_case"].items()
    }
    for name, _ in rules:
        fixed, lost = _gainers(results[name]["per_case"], reference)
        print(f"  vs calibrated: {name:<34} fixed={len(fixed)} lost={len(lost)} {lost or ''}")

    if args.detail:
        chosen = results.get(args.detail)
        if chosen:
            fixed, lost = _gainers(chosen["per_case"], reference)
            print(f"\n{args.detail}: fixed={fixed}")
            print(f"{args.detail}: lost={lost}")

    out = Path(args.data).with_name("fusion_rule_sweep.json")
    out.write_text(
        json.dumps(
            {
                name: {k: v for k, v in result.items() if k != "per_case"}
                for name, result in results.items()
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()