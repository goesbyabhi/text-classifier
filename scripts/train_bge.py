"""
Fit calibration on the REAL frozen bge embedder via Workers AI, then judge on
the eval holdout. This replaces the hash-v0 stand-in numbers that local-only
training produced: script/train.py stays for offline work, this is the one
that makes the deployed confidence numbers real.

Embeds every unique text and label once through
    POST /accounts/{account}/ai/run/@cf/baai/bge-small-en-v1.5
batched (Workers AI accepts arrays). Auth is the wrangler OAuth token read
from the wrangler config; no secrets handled in the repo.

Usage:
    uv run python scripts/train_bge.py --account <id> [--out src/calibration.py]
"""

from __future__ import annotations

import argparse
import json
import tomllib
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark import render as render_benchmark  # noqa: E402
from classifier import DEFAULT_CALIBRATION, Scorer, fit_multi, fit_multi_threshold, fit_temperature, sigmoid, softmax_rows  # noqa: E402

MODEL = "@cf/baai/bge-small-en-v1.5"


def oauth_token():
    cfg = Path.home() / "AppData" / "Roaming" / "xdg.config" / ".wrangler" / "config" / "default.toml"
    with open(cfg, "rb") as f:
        data = tomllib.load(f)
    return data.get("oauth_token")


def embed_texts(account: str, token: str, texts, batch: int = 32):
    """Embed texts via Workers AI REST, mean-normalising nothing (Scorer does it)."""
    rows = []
    for i in range(0, len(texts), batch):
        chunk = list(texts[i : i + batch])
        body = json.dumps({"text": chunk}).encode("utf-8")
        req = urllib.request.Request(
            f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{MODEL}",
            data=body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as res:
            reply = json.loads(res.read().decode("utf-8"))
        rows.extend(reply["result"]["data"])
    return np.asarray(rows, dtype=np.float32)


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
        ls = r.get("labels") if r.get("labels") else ([r["label"]] if r.get("label") else [])
        for l in ls:
            if l not in labels:
                labels.append(l)
    return labels


def emb_and_scores(records, labels, account, token, embed_cache):
    to_embed = list(labels) + [r["text"] for r in records]
    fresh = [t for t in to_embed if t not in embed_cache]
    if fresh:
        vectors = embed_texts(account, token, fresh)
        embed_cache.update(zip(fresh, vectors))
    label_embs = np.asarray([embed_cache[l] for l in labels])
    text_embs = np.asarray([embed_cache[r["text"]] for r in records])
    return Scorer().scores(text_embs, label_embs)


def accuracy(scores, records, labels):
    idx = np.argmax(scores, axis=1)
    hits = sum(1 for i, r in enumerate(records) if r.get("label") and labels[idx[i]] == r["label"])
    return hits / sum(1 for r in records if r.get("label"))


def macro_f1(scores, records, labels, bias, mtemp, threshold):
    y_pred = (sigmoid((scores - bias) / mtemp) >= threshold).astype(int)
    tp = fp = fn = 0
    for i, r in enumerate(records):
        true = set(r.get("labels") or [])
        picked = {labels[j] for j in range(len(labels)) if y_pred[i, j]}
        for l in true:
            if l in picked: tp += 1
            else: fn += 1
        for l in picked - true:
            fp += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def reliability(scores, records, labels, temperature):
    """Proportion where the true label's softmax confidence exceeds 0.5."""
    p = softmax_rows(scores / temperature)
    hits = 0
    n = 0
    for i, r in enumerate(records):
        if not r.get("label"):
            continue
        j = labels.index(r["label"])
        hits += 1 if p[i, j] > 0.5 else 0
        n += 1
    return hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path, nargs="?", default=Path("eval/data/calibration.jsonl"))
    ap.add_argument("--eval", type=Path, default=Path("eval/data/eval.jsonl"))
    ap.add_argument("--account", required=True)
    ap.add_argument("--out", type=Path, default=Path("src/calibration.py"))
    args = ap.parse_args()

    token = oauth_token()
    if not token:
        raise SystemExit("no OAuth token in wrangler config; run `uv run pywrangler login` first.")

    train_records = load_records(args.dataset)
    eval_records = load_records(args.eval) if args.eval.exists() else []
    train_labels = label_set(train_records)
    eval_labels = label_set(eval_records)
    all_labels = []
    for l in train_labels + eval_labels:
        if l not in all_labels:
            all_labels.append(l)

    print(f"embedding {len(all_labels)} labels + {len(train_records)} train + {len(eval_records)} eval texts via {MODEL}")
    cache = {}
    train_scores = emb_and_scores(train_records, all_labels, args.account, token, cache)
    print(f"train matrix {train_scores.shape}")

    temperature = DEFAULT_CALIBRATION["temperature"]
    nll = None
    single_rows = [i for i, r in enumerate(train_records) if r.get("label")]
    if single_rows:
        ts = train_scores[np.array(single_rows)]
        true_idx = np.array([all_labels.index(train_records[i]["label"]) for i in single_rows])
        temperature, nll = fit_temperature(ts, true_idx)
        print(f"single-label: temperature={temperature:.4f} nll={nll:.4f} n={len(single_rows)}")

    bias = DEFAULT_CALIBRATION["multi_bias"]
    mtemp = DEFAULT_CALIBRATION["multi_temperature"]
    threshold = DEFAULT_CALIBRATION["multi_threshold"]
    bce = f1 = None
    multi_rows = [i for i, r in enumerate(train_records) if r.get("labels")]
    if multi_rows:
        y = np.zeros((len(multi_rows), len(all_labels)), dtype=float)
        for k, i in enumerate(multi_rows):
            for l in train_records[i]["labels"]:
                y[k, all_labels.index(l)] = 1.0
        bias, mtemp, bce = fit_multi(train_scores[np.array(multi_rows)], y)
        threshold, f1 = fit_multi_threshold(train_scores[np.array(multi_rows)], y, bias, mtemp)
        print(f"multi-label: bias={bias:.4f} temperature={mtemp:.4f} threshold={threshold:.4f} bce={bce:.4f} f1={f1:.4f} n={len(multi_rows)}")

    cal = {
        "temperature": round(temperature, 4),
        "multi_bias": round(bias, 4),
        "multi_temperature": round(mtemp, 4),
        "multi_threshold": round(threshold, 4),
        "fitted_on": date.today().isoformat(),
        "dataset": args.dataset.name,
        "embedder": MODEL,
    }

    if eval_records:
        ev_scores = emb_and_scores(eval_records, all_labels, args.account, token, cache)
        acc = accuracy(ev_scores, eval_records, all_labels)
        mf1 = macro_f1(ev_scores, eval_records, all_labels, bias, mtemp, threshold)
        rel = reliability(ev_scores, eval_records, all_labels, temperature)
        report = {
            "name": "eval-bge",
            "dataset": args.eval.name,
            "date": cal["fitted_on"],
            "embedder": MODEL,
            "calibration": cal,
            "single_accuracy": acc,
            "multi_f1": round(mf1, 4),
            "reliability": round(rel, 4),
            "n": len(eval_records),
        }
        out_dir = Path("eval/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval-bge.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"eval holdout: single_accuracy={acc:.2f} multi_f1={mf1:.4f} reliability(acc>0.5)={rel:.2f} n={len(eval_records)}")
        print(f"wrote eval/results/eval-bge.json")

        benchmark_text = render_benchmark(report)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        (args.out.parent / "benchmark_data.py").write_text(
            '"""Generated by scripts/train_bge.py - do not edit by hand."""\n\n'
            f"BENCHMARK = {json.dumps(benchmark_text)}\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out.parent / 'benchmark_data.py'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f'    {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},' for k, v in cal.items())
    args.out.write_text(
        '"""Generated by scripts/train_bge.py - do not edit by hand."""\n\n'
        "CALIBRATION = {\n"
        f"{body}\n"
        "}\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()