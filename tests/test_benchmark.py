import json
import sys
from pathlib import Path

from benchmark import STUB, get_benchmark, render


def _report(**overrides):
    report = {
        "name": "eval-bge",
        "dataset": "eval.jsonl",
        "date": "2026-09-20",
        "embedder": "@cf/baai/bge-small-en-v1.5",
        "calibration": {
            "temperature": 0.1,
            "multi_bias": 1.5356,
            "multi_temperature": 0.5477,
            "multi_threshold": 0.1423,
        },
        "single_accuracy": 0.7,
        "multi_f1": 0.0816,
        "reliability": 0.4,
        "n": 15,
    }
    report.update(overrides)
    return report


def test_render_starts_with_header():
    text = render(_report())
    assert text.startswith("text-classifier — measured benchmark")


def test_render_carries_the_measured_numbers():
    text = render(_report())
    assert "single-label accuracy   0.7000" in text
    assert "multi-label F1          0.0816 @ threshold 0.1423" in text
    assert "reliability (top>0.5)   0.4000" in text
    assert "temperature       0.1000" in text
    assert "multi_threshold   0.1423" in text


def test_render_drops_missing_sections():
    text = render(_report(single_accuracy=None, reliability=None, multi_f1=None))
    assert "single-label accuracy" not in text
    assert "multi-label F1" not in text
    assert "reliability (top>0.5)" not in text
    text = render(_report(calibration={}, single_accuracy=None, reliability=None, multi_f1=None))
    assert "temperature" not in text
    assert "multi_threshold" not in text


def test_render_with_report_from_disk():
    report = json.loads(Path("eval/results/eval-bge.json").read_text(encoding="utf-8"))
    text = render(report)
    assert f'({report["n"]} records)' in text


def test_stub_is_honest_when_nothing_measured():
    assert "No measured benchmark" in STUB


def test_get_benchmark_falls_back_to_stub_when_module_absent(monkeypatch):
    def boom(name, *args, **kwargs):
        raise ImportError(f"no module {name}")

    monkeypatch.setattr("builtins.__import__", boom)
    assert get_benchmark() == STUB


def test_get_benchmark_returns_generated_text(monkeypatch):
    class _B:
        BENCHMARK = render(_report())

    monkeypatch.setitem(sys.modules, "benchmark_data", _B)
    assert get_benchmark() == _B.BENCHMARK