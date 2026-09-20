# eval/

Benchmarks are the ONLY source of measured numbers in this repository. A
number belongs in `AGENTS.md`, the docs, or the home page only if it was
produced here. Never transcribe a measurement by hand.

## How a number is measured

Every report follows the same recipe, so two reports are comparable:

1. **Data** – a labeled JSONL set under `eval/data/` (gitignored; reload
   scripts live here too once real sets are acquired).
2. **Pipeline** – the identical math the Worker runs: embeddings (the
   `HashEmbedder` stand-in for offline `eval/bench.py`; the real bge
   embedder via Workers AI for `scripts/train_bge.py`) feeding
   `classifier.Scorer`, calibrated from `src/calibration.py` if present
   (and its `embedder` matches the one scoring), defaults otherwise.
3. **Report** – written to `eval/results/<name>.json` (gitignored) and
   printed. The JSON pins the dataset name, the date, the embedder, the
   calibration it ran under, the wall-clock ms, accuracy (single-label) and
   F1 (multi-label).

```sh
uv run python eval/bench.py eval/data/calibration.jsonl --name calibration
```

## The property that matters

Accuracy is necessary but not sufficient: a classifier that quietly says 0.99
when it has no idea is worse than one that says 0.55 correctly. The
`classifier.py` fits a temperature so the confidence IS the observed
frequency. Judge the math on **calibration** as well as raw accuracy:

- temperature-scaled softmax: reported confidence vs. actual hit rate at the
  same confidence, over the holdout;
- multi-label sigmoid: precision/recall/F1 against the threshold.

## Layout

- `bench.py` – runs the harness, writes the report
- `data/` – labeled sets (gitignored)
- `results/` – measured reports (gitignored)