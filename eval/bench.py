"""
Benchmark harness. The ONLY source of measured numbers in this repo.

Runs the offline pipeline (HashEmbedder -> Scorer, exactly the math the
Worker runs, with a calibration file if one exists) against a labeled eval
set, and writes a JSON report under eval/results/. A measured number belongs
in docs only if it came from here.

Dataset format (JSONL):
    {"text": "...", "label": "spam"}              single-label eval record
    {"text": "...", "labels": ["a", "b"]}         multi-label eval record

Usage:
    uv run python eval/bench.py eval/data/eval.jsonl --name myset

A report is always written to eval/results/<name>.json (gitignored) and the
key numbers printed. Never transcribe a number by hand; read it from the
report or the printed JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classifier import Scorer, load_calibration  # noqa: E402
from embeddings import HashEmbedder  # noqa: E402


def load_records(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def label_set(records):
    labels = []
    for r in records:
        ls = r.get("labels") or ([r["label"]] if r.get("label") else [])
        for l in ls:
            if l not in labels:
                labels.append(l)
    return labels


def evaluate(records, labels, scorer, embedder):
    texts = [r["text"] for r in records]
    single_hits = 0
    single_total = 0
    multi_rows = []
    for i, r in enumerate(records):
        if r.get("labels"):
            multi_rows.append((i, set(r["labels"])))
        elif r.get("label"):
            single_total += 1

    label_embs = embedder.embed_texts(labels)
    text_embs = embedder.embed_texts(texts)

    if single_total:
        probs = scorer.single_probs(text_embs, label_embs)
        idx = [i for i, r in enumerate(records) if r.get("label")]
        for i in idx:
            if labels[int(probs[i].argmax())] == records[i]["label"]:
                single_hits += 1
    single_acc = single_hits / single_total if single_total else None

    tp = fp = fn = 0
    if multi_rows:
        probs = scorer.multi_probs(text_embs, label_embs)
        for i, true_set in multi_rows:
            picked = {labels[j] for j in range(len(labels)) if probs[i, j] >= scorer.multi_threshold}
            tp += len(true_set & picked)
            fp += len(picked - true_set)
            fn += len(true_set - picked)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None

    return {
        "n": len(records),
        "single_accuracy": single_acc,
        "multi_f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path, help="JSONL eval set")
    ap.add_argument("--name", default=None, help="report name; defaults to dataset stem")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="explicit JSON calibration file (default: bundled src/calibration.py)")
    args = ap.parse_args()

    from embeddings import HashEmbedder as _HE

    records = load_records(args.dataset)
    labels = label_set(records)
    embedder = _HE()
    scorer = Scorer(
        load_calibration(
            str(args.calibration) if args.calibration else None,
            embedder=embedder.name,
        )
    )

    started = time.time()
    result = evaluate(records, labels, scorer, embedder)
    ms = (time.time() - started) * 1000

    name = args.name or args.dataset.stem
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "name": name,
        "dataset": args.dataset.name,
        "date": time.strftime("%Y-%m-%d"),
        "embedder": embedder.name,
        "calibration": scorer.temperature,  # the calibration that scored this
        "ms": round(ms, 1),
        **result,
    }
    out = out_dir / f"{name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()