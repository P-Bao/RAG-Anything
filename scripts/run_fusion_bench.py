#!/usr/bin/env python3
"""Benchmark per-modality fusion strategies against the live retrieval test set.

For each strategy the script recreates the API container with that strategy's
`RETRIEVAL_FUSION_*` env, waits for readiness, then replays every case in
`tests/retrieval_cases.json` through `POST /v2/rag` and records Hit@5, MRR,
per-case outcome and latency. Scoring is imported from the live pytest module so
the numbers here match the official assertions in
`tests/test_retrieval_recall_live.py`.

Usage:
    python scripts/run_fusion_bench.py                  # full matrix
    python scripts/run_fusion_bench.py --only quota,rrf_k10
    python scripts/run_fusion_bench.py --skip-single    # matrix minus baseline
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx

from tests.test_retrieval_recall_live import BASE_URL, run_case

COMPOSE_FILE = REPO_ROOT / "docker-compose.ami.yml"
SERVICE = "ami-rag-api"
MODALITIES = ("text", "table", "image")

POOL_SIZES_DEFAULT = "text=50,image=15"
VL_POOLS_DEFAULT = "text,image"
POOLS_DEFAULT = "text+table,image"
QUOTA_DEFAULT = "text=3@0.45,image=2@0.40"

# name -> RETRIEVAL_FUSION_* overrides. The first entry is the pre-fusion
# baseline the rest are compared against.
STRATEGIES: dict[str, dict[str, str]] = {
    "single": {},
    # The recommended configuration: two branches, tables kept as images in the
    # shared rerank, merged text pool deep enough for table_08.
    "calibrated": {"RETRIEVAL_FUSION_MODE": "calibrated"},
    # Each knob isolated against the recommended row.
    "calibrated_gate_off": {
        "RETRIEVAL_FUSION_MODE": "calibrated",
        "RETRIEVAL_FUSION_IMAGE_GATE": "false",
    },
    "calibrated_text_only_tables": {
        "RETRIEVAL_FUSION_MODE": "calibrated",
        "RETRIEVAL_FUSION_VL_POOLS": "image",
    },
    "calibrated_text40": {
        "RETRIEVAL_FUSION_MODE": "calibrated",
        "RETRIEVAL_FUSION_POOL_SIZES": "text=40,image=15",
    },
    "calibrated_text60": {
        "RETRIEVAL_FUSION_MODE": "calibrated",
        "RETRIEVAL_FUSION_POOL_SIZES": "text=60,image=15",
    },
    "calibrated_image30": {
        "RETRIEVAL_FUSION_MODE": "calibrated",
        "RETRIEVAL_FUSION_POOL_SIZES": "text=50,image=30",
    },
    "rrf_k60": {"RETRIEVAL_FUSION_MODE": "rrf", "RETRIEVAL_FUSION_RRF_K": "60"},
    "rrf_k10": {"RETRIEVAL_FUSION_MODE": "rrf", "RETRIEVAL_FUSION_RRF_K": "10"},
    "quota_default": {"RETRIEVAL_FUSION_MODE": "quota", "RETRIEVAL_FUSION_QUOTA": QUOTA_DEFAULT},
    "quota_no_floor": {
        "RETRIEVAL_FUSION_MODE": "quota",
        "RETRIEVAL_FUSION_QUOTA": "text=3,image=2",
    },
    "no_calibration": {
        "RETRIEVAL_FUSION_MODE": "calibrated",
        "RERANK_CALIBRATION_ENABLED": "false",
    },
}


def _deploy(overrides: dict[str, str], build: bool) -> None:
    env = dict(os.environ)
    env.update(overrides)
    env.setdefault("RETRIEVAL_FUSION_POOLS", POOLS_DEFAULT)
    env.setdefault("RETRIEVAL_FUSION_POOL_SIZES", POOL_SIZES_DEFAULT)
    env.setdefault("RETRIEVAL_FUSION_QUOTA", QUOTA_DEFAULT)
    env.setdefault("RETRIEVAL_FUSION_VL_POOLS", VL_POOLS_DEFAULT)
    # The baseline runs the un-fused flow, so make sure nothing in the ambient
    # environment leaks a fusion mode into it.
    if not overrides.get("RETRIEVAL_FUSION_MODE"):
        env["RETRIEVAL_FUSION_MODE"] = "single"
    args = ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d"]
    if build:
        args.append("--build")
    args.append(SERVICE)
    subprocess.run(args, cwd=REPO_ROOT, env=env, check=True)


def _wait_ready(timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{BASE_URL}/healthz", timeout=5.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass  # container still starting
        time.sleep(2.0)
    return False


def _p50(values: list[float]) -> int:
    return int(statistics.median(values)) if values else 0


def _score(results: list[dict], cases: list[dict]) -> dict:
    """Aggregate `run_case` output into the numbers the report prints.

    Relies on the fields `run_case` already computes, so the benchmark and the
    official pytest assertions can never disagree about what a hit is.
    """
    by_modality = {m: {"hit": 0, "total": 0} for m in MODALITIES}
    rr_sum = 0.0
    hits = 0
    client_ms, server_ms = [], []
    case_rows: dict[str, dict] = {}

    for result in results:
        modality = result["modality"]
        bucket = by_modality.setdefault(modality, {"hit": 0, "total": 0})
        bucket["total"] += 1

        hit = result["status"] == "hit"
        if hit:
            hits += 1
            bucket["hit"] += 1
            rr_sum += 1.0 / max(result["rank"], 1)

        case_rows[result["id"]] = {"status": result["status"], "modality": modality}
        client_ms.append(result["latency_ms"])
        if result.get("server_latency_ms"):
            server_ms.append(result["server_latency_ms"])

    total = len(results)
    return {
        "hit_at_5": hits,
        "total": total,
        "mrr": rr_sum / total if total else 0.0,
        "by_modality": by_modality,
        "latency_ms": {
            "client_p50": _p50(client_ms),
            "server_p50": _p50(server_ms),
        },
        "cases": case_rows,
    }


_FUSION_ENV = (
    "RETRIEVAL_FUSION_MODE",
    "RETRIEVAL_FUSION_POOLS",
    "RETRIEVAL_FUSION_POOL_SIZES",
    "RETRIEVAL_FUSION_VL_POOLS",
    "RETRIEVAL_FUSION_IMAGE_GATE",
    "RETRIEVAL_FUSION_RRF_K",
    "RETRIEVAL_FUSION_QUOTA",
    "RERANK_CALIBRATION_ENABLED",
)


def _assert_deployed(overrides: dict[str, str]) -> None:
    """Fail loudly if the container is not actually running the strategy.

    Compose interpolation has silently served a stale default here more than once,
    which makes a strategy row look like a real regression instead of a config bug.
    """
    actual = subprocess.run(
        ["docker", "exec", SERVICE, "printenv", *_FUSION_ENV],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    deployed = dict(zip(_FUSION_ENV, actual))
    for key, want in overrides.items():
        got = deployed.get(key)
        if got != want:
            raise SystemExit(
                f"container has {key}={got!r} but strategy asked for {want!r}; "
                f"full env: {deployed}"
            )


def _run_strategy(name: str, overrides: dict[str, str], cases: list[dict], build: bool) -> dict:
    print(f"\n=== {name} :: {overrides or 'baseline (single pool)'} ===", flush=True)
    _deploy(overrides, build)
    if not _wait_ready():
        raise SystemExit(f"API not ready for strategy {name}")
    _assert_deployed(overrides)
    # First request pays lazy init (pipeline, asset store, calibrator).
    run_case(cases[0])

    results = [run_case(case) for case in cases]
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        print(f"  !! {len(errors)} case(s) returned an API error, first: {errors[0]['detail']}")
    scored = _score(results, cases)
    print(
        f"  Hit@5 {scored['hit_at_5']}/{scored['total']} "
        f"(p50 {scored['latency_ms']['client_p50']} ms, server p50 "
        f"{scored['latency_ms']['server_p50']} ms)",
        flush=True,
    )
    return {"strategy": name, "env": overrides, **scored}


def _print_report(rows: list[dict], baseline: dict | None) -> None:
    header = (
        f"| {'strategy':<22} | {'HIT@5':<9} | {'text':<7} | {'table':<7} | "
        f"{'image':<7} | {'MRR':<7} | {'p50 ms':<7} | {'vs base':<7} |"
    )
    print("\n" + "=" * 92)
    print("FUSION STRATEGY BENCHMARK".center(92))
    print("=" * 92)
    print(header)
    print("-" * 92)
    for row in rows:
        mods = row["by_modality"]
        delta = "" if baseline is None else f"{row['hit_at_5'] - baseline['hit_at_5']:+d}"
        print(
            f"| {row['strategy']:<22} | {row['hit_at_5']:>2}/{row['total']:<6} | "
            f"{mods['text']['hit']:>2}/{mods['text']['total']:<4} | "
            f"{mods['table']['hit']:>2}/{mods['table']['total']:<4} | "
            f"{mods['image']['hit']:>2}/{mods['image']['total']:<4} | "
            f"{row['mrr']:<7.4f} | {row['latency_ms']['client_p50']!s:<7} | {delta:<7} |"
        )
    print("-" * 92)

    if baseline:
        fixed: dict[str, list[str]] = {}
        broke: dict[str, list[str]] = {}
        for row in rows:
            for cid, case in baseline["cases"].items():
                after = row["cases"].get(cid, {}).get("status")
                if case["status"] == "miss" and after == "hit":
                    fixed.setdefault(row["strategy"], []).append(cid)
                elif case["status"] == "hit" and after == "miss":
                    broke.setdefault(row["strategy"], []).append(cid)
        for row in rows:
            name = row["strategy"]
            print(f"  {name:<22} fixed: {', '.join(fixed.get(name, [])) or '-'}")
            print(f"  {name:<22} broke: {', '.join(broke.get(name, [])) or '-'}")
    print("=" * 92 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(REPO_ROOT / "tests" / "retrieval_cases.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "bench_fusion_results.json"))
    parser.add_argument("--only", help="comma-separated subset of strategies to run")
    parser.add_argument("--skip-single", action="store_true", help="do not run the baseline row")
    args = parser.parse_args()

    names = list(STRATEGIES)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in STRATEGIES]
        if unknown:
            raise SystemExit(f"Unknown strategies: {', '.join(unknown)}")
    if args.skip_single:
        names = [n for n in names if n != "single"]

    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)
    print(f"Benchmarking {len(names)} strategies over {len(cases)} cases at {BASE_URL}")

    rows = []
    for index, name in enumerate(names):
        # Only the first deploy needs a rebuild; later ones differ by env alone.
        rows.append(_run_strategy(name, STRATEGIES[name], cases, build=(index == 0)))

    baseline = next((r for r in rows if r["strategy"] == "single"), None)
    _print_report(rows, baseline)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"base_url": BASE_URL, "results": rows}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()