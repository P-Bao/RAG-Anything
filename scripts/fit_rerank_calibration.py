#!/usr/bin/env python3
"""Offline fitting script for per-modality rerank score calibration.

Collects score samples from the live API / vector store, fits Platt scaling
(and optionally Isotonic regression) models per modality, evaluates via
4-fold cross-validation, and writes `rerank_calibration.json`.
"""

import argparse
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ami_rag.core.calibration import ModalityCalibration, RerankCalibrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fit_calibration")


def load_cases(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_data_from_api(
    cases: list[dict],
    base_url: str = "http://localhost:8009",
    top_k: int = 20,
) -> list[dict]:
    """Collect (score, modality, label, query_id) samples by querying the API and image store."""
    samples = []
    client = httpx.Client(timeout=60.0)

    logger.info("Collecting candidate samples for %d test cases from API...", len(cases))

    for case in cases:
        cid = case["id"]
        modality = case["modality"]
        query = case["query"]

        try:
            resp = client.post(
                f"{base_url}/v2/rag/",
                json={"messages": [{"role": "user", "content": query}], "top_k": top_k},
            )
            if resp.status_code != 200:
                logger.warning("[%s] API returned %s: %s", cid, resp.status_code, resp.text[:100])
                continue
            data = resp.json()
            docs = data.get("documents", [])

            for doc in docs:
                # Phải lấy điểm rerank THÔ. `doc["score"]` là điểm đã qua
                # calibration (khi fusion/single bật calibrator) hoặc điểm đã
                # fuse, nên dùng nó sẽ fit calibrator trên output của chính nó.
                metadata = doc.get("metadata") or {}
                raw_score = metadata.get("raw_score")
                has_raw = raw_score is not None
                if raw_score is None:
                    raw_score = doc.get("score")
                if raw_score is None:
                    continue
                doc_mod = doc.get("modality") or "text"

                # Ground truth labeling
                label = 0
                if modality == "text" and doc_mod == "text":
                    doc_id = (doc.get("doc") or {}).get("document_id")
                    if doc_id == case["expected_doc_id"]:
                        label = 1
                elif modality in ("table", "image") and doc_mod == modality:
                    ref_id = doc.get("reference_id")
                    if ref_id == case["expected_chunk_id"]:
                        label = 1

                samples.append({
                    "case_id": cid,
                    "case_modality": modality,
                    "sample_modality": doc_mod,
                    "score": float(raw_score),
                    "has_raw": has_raw,
                    "label": int(label),
                })
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("[%s] Request failed: %s", cid, exc)

    logger.info("Collected %d samples from standard API requests", len(samples))
    return samples


def supplement_image_samples(cases: list[dict], existing_samples: list[dict]) -> list[dict]:
    """Ensure sufficient positive and negative image scores by checking Qdrant image pool if needed."""
    image_cases = [c for c in cases if c["modality"] == "image"]
    pos_count = sum(1 for s in existing_samples if s["sample_modality"] == "image" and s["label"] == 1)

    logger.info("Current image positive samples in standard pool: %d", pos_count)
    if pos_count >= 3:
        return existing_samples

    # If images were pushed out of top 20 by text, fetch directly from docker environment
    logger.info("Querying image pool in container to collect calibrated scores for image modality...")

    py_script = """
import os, json, httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from ami_rag.core.factory import get_asset_store
from ami_rag.core.rerank_client import build_rerank_documents, build_vllm_rerank_func
import asyncio

cases = json.loads('''__CASES_JSON__''')
client = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ.get('QDRANT_API_KEY'), check_compatibility=False)
col = 'multimodal__llama-nemotron-embed-vl-1b-v2__v1'
embed_url = os.environ.get('EMBED_SERVER_URL') + '/v1/embeddings'
embed_model = os.environ.get('EMBED_MODEL')

async def run():
    out = []
    rf = build_vllm_rerank_func()
    asset_store = get_asset_store()
    for c in cases:
        resp = httpx.post(embed_url, json={'input': [c['query']], 'model': embed_model, 'input_type': 'query'}, timeout=30.0)
        vec = resp.json()['data'][0]['embedding']
        results = client.query_points(
            collection_name=col,
            query=vec,
            query_filter=Filter(must=[FieldCondition(key='modality', match=MatchValue(value='image'))]),
            limit=10,
            with_payload=True
        ).points
        chunks = [h.payload for h in results]
        docs = await build_rerank_documents(chunks, asset_store, multimodal=True)
        res = await rf(query=c['query'], documents=docs, top_n=10)
        for r in res:
            chunk = chunks[r['index']]
            is_target = chunk.get('chunk_id') == c['expected_chunk_id']
            out.append({
                'case_id': c['id'],
                'case_modality': 'image',
                'sample_modality': 'image',
                'score': float(r['relevance_score']),
                'label': 1 if is_target else 0
            })
    print('RESULT_JSON_START')
    print(json.dumps(out))

asyncio.run(run())
"""
    cmd = [
        "docker", "exec", "ami-rag-api", "python", "-c",
        py_script.replace("__CASES_JSON__", json.dumps(image_cases)),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = proc.stdout.split("RESULT_JSON_START")
        if len(parts) > 1:
            img_samples = json.loads(parts[1].strip())
            logger.info("Retrieved %d image samples directly from container reranker", len(img_samples))
            existing_samples.extend(img_samples)
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Container query for image samples failed: %s", exc)

    return existing_samples


def fit_platt_model(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling (1D Logistic Regression with balanced weighting)."""
    X = scores.reshape(-1, 1)
    y = labels

    # If only one class is present, provide fallback heuristic parameters
    if len(np.unique(y)) < 2:
        return 10.0, -1.0

    clf = LogisticRegression(C=1000.0, class_weight="balanced", solver="lbfgs")
    clf.fit(X, y)
    a = float(clf.coef_[0][0])
    b = float(clf.intercept_[0])
    return a, b


def fit_isotonic_model(scores: np.ndarray, labels: np.ndarray) -> tuple[list[float], list[float]]:
    """Fit Isotonic Regression with clipping."""
    if len(np.unique(labels)) < 2:
        return [0.0, 1.0], [0.0, 1.0]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, labels)
    return [float(x) for x in iso.X_thresholds_], [float(y) for y in iso.y_thresholds_]


def evaluate_cross_validation(samples_by_modality: dict[str, list[dict]], method: str = "platt", k: int = 4):
    """Run K-Fold Cross Validation across queries to measure Brier Score reduction."""
    logger.info("--- %d-FOLD CROSS VALIDATION EVALUATION (%s) ---", k, method.upper())

    for mod, samps in samples_by_modality.items():
        # Group by case_id to prevent data leakage across samples of the same query
        cases = list({s["case_id"] for s in samps})
        if len(cases) < k:
            logger.info("Modality %s has only %d queries, skipping CV", mod, len(cases))
            continue

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        raw_briers, cal_briers = [], []

        for train_case_idx, test_case_idx in kf.split(cases):
            train_cases = {cases[i] for i in train_case_idx}
            test_cases = {cases[i] for i in test_case_idx}

            train_s = [s for s in samps if s["case_id"] in train_cases]
            test_s = [s for s in samps if s["case_id"] in test_cases]

            if not train_s or not test_s:
                continue

            X_tr = np.array([s["score"] for s in train_s])
            y_tr = np.array([s["label"] for s in train_s])
            X_te = np.array([s["score"] for s in test_s])
            y_te = np.array([s["label"] for s in test_s])

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            # Raw brier (using score clipped to 0-1 as probability proxy)
            raw_brier = brier_score_loss(y_te, np.clip(X_te, 0.0, 1.0))
            raw_briers.append(raw_brier)

            if method == "platt":
                a, b = fit_platt_model(X_tr, y_tr)
                p_pred = 1.0 / (1.0 + np.exp(-(a * X_te + b)))
            else:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(X_tr, y_tr)
                p_pred = iso.predict(X_te)

            cal_brier = brier_score_loss(y_te, p_pred)
            cal_briers.append(cal_brier)

        if cal_briers:
            logger.info(
                "Modality: %-6s | Raw Brier: %.4f -> Calibrated Brier: %.4f (improvement: %.1f%%)",
                mod.upper(),
                float(np.mean(raw_briers)),
                float(np.mean(cal_briers)),
                float((np.mean(raw_briers) - np.mean(cal_briers)) / np.mean(raw_briers) * 100),
            )


def main():
    parser = argparse.ArgumentParser(description="Fit per-modality rerank calibrator")
    parser.add_argument("--cases", default=str(REPO_ROOT / "tests" / "retrieval_cases.json"))
    parser.add_argument("--output", default=str(REPO_ROOT / "rerank_calibration.json"))
    parser.add_argument("--method", choices=["platt", "isotonic"], default="platt")
    parser.add_argument("--cv", type=int, default=4)
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    samples = collect_data_from_api(cases)
    samples = supplement_image_samples(cases, samples)

    samples_by_modality = defaultdict(list)
    for s in samples:
        samples_by_modality[s["sample_modality"]].append(s)

    # Print data summary
    print("\n" + "=" * 60)
    print("           SAMPLE COLLECTION SUMMARY BY MODALITY")
    print("=" * 60)
    for mod, samps in samples_by_modality.items():
        pos = sum(1 for s in samps if s["label"] == 1)
        neg = len(samps) - pos
        scores = [s["score"] for s in samps]
        no_raw = sum(1 for s in samps if not s.get("has_raw"))
        print(f"Modality: {mod.upper():<6} | Total: {len(samps):<4} (Pos: {pos:<2}, Neg: {neg:<3}) | Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        if no_raw:
            print(f"         ^ cảnh báo: {no_raw}/{len(samps)} mẫu KHÔNG có metadata.raw_score -> lấy doc['score'] (có thể đã calibration)")
    print("=" * 60 + "\n")

    # Evaluate CV
    evaluate_cross_validation(samples_by_modality, method=args.method, k=args.cv)

    # Fit final models on all data
    models = {}
    for mod, samps in samples_by_modality.items():
        X = np.array([s["score"] for s in samps])
        y = np.array([s["label"] for s in samps])
        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))

        if args.method == "platt":
            a, b = fit_platt_model(X, y)
            models[mod] = ModalityCalibration(method="platt", a=a, b=b, samples_pos=pos, samples_neg=neg)
            logger.info("Fitted Platt for %s: a=%.4f, b=%.4f", mod, a, b)
        else:
            xs, ys = fit_isotonic_model(X, y)
            models[mod] = ModalityCalibration(method="isotonic", x=xs, y=ys, samples_pos=pos, samples_neg=neg)
            logger.info("Fitted Isotonic for %s: %d thresholds", mod, len(xs))

    # Also fit default model on all pooled samples
    all_X = np.array([s["score"] for s in samples])
    all_y = np.array([s["label"] for s in samples])
    if args.method == "platt":
        da, db = fit_platt_model(all_X, all_y)
        models["default"] = ModalityCalibration(method="platt", a=da, b=db, samples_pos=int(np.sum(all_y == 1)), samples_neg=int(np.sum(all_y == 0)))
    else:
        dxs, dys = fit_isotonic_model(all_X, all_y)
        models["default"] = ModalityCalibration(method="isotonic", x=dxs, y=dys, samples_pos=int(np.sum(all_y == 1)), samples_neg=int(np.sum(all_y == 0)))

    calibrator = RerankCalibrator(
        models=models,
        version="1.0",
        metadata={
            "method": args.method,
            "num_cases": len(cases),
            "num_samples": len(samples),
        },
    )
    calibrator.save(args.output)
    logger.info("Saved calibration models to %s", args.output)


if __name__ == "__main__":
    main()
