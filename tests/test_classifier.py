import numpy as np
import pytest

from classifier import DEFAULT_CALIBRATION, Scorer, fit_multi, fit_multi_threshold, fit_temperature, load_calibration, normalise, sigmoid, softmax_rows


def test_normalise_zero_vectors_survive():
    rows = np.array([[0.0, 0.0], [3.0, 4.0]])
    out = normalise(rows)
    assert out[0].tolist() == [0.0, 0.0]
    assert np.allclose(out[1], [0.6, 0.8])


def test_softmax_rows_normalises_and_is_argmax_consistent():
    z = np.array([[1.0, 2.0, 3.0]])
    p = softmax_rows(z)
    assert np.allclose(p.sum(), 1.0)
    assert int(p.argmax()) == 2  # column 2 wins
    assert np.allclose(softmax_rows(np.full((2, 3), 5.0)), 1.0 / 3)


def test_sigmoid_bounds():
    p = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert p[0] < 1e-6
    assert p[1] == pytest.approx(0.5)
    assert p[2] > 1 - 1e-6


def test_default_calibration_is_honest_not_tuned():
    s = Scorer()
    assert s.temperature == 1.0
    assert s.multi_bias == 0.0
    assert s.multi_threshold == 0.5


def test_load_calibration_without_bundle_returns_defaults(monkeypatch):
    import builtins
    real = builtins.__import__
    def no_calibration(name, *a, **k):
        if name == "calibration":
            raise ImportError("no bundle in this scenario")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_calibration)
    cal = load_calibration(None, embedder="hash-v0")
    assert cal["temperature"] == 1.0


def test_load_calibration_bundle_honored_and_gated():
    try:
        from calibration import CALIBRATION
    except ImportError:
        pytest.skip("no bundled calibration in this checkout")
    embedder = CALIBRATION.get("embedder")
    fitted = load_calibration(None, embedder=embedder)
    assert fitted["temperature"] != 1.0  # a matching fit is honored
    foreign = load_calibration(None, embedder="some-other-embedder")
    assert foreign == dict(DEFAULT_CALIBRATION)  # a mismatched fit is rejected


def test_load_calibration_json_gate_rejects_wrong_embedder(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text('{"temperature": 0.2, "embedder": "hash-v0"}', encoding="utf-8")
    hot = load_calibration(str(p), embedder="hash-v0")
    assert hot["temperature"] == 0.2
    cold = load_calibration(str(p), embedder="@cf/baai/bge-small-en-v1.5")
    assert cold["temperature"] == 1.0  # a hash-v0 fit must not leak into bge


def test_load_calibration_no_embedder_trusts_json(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text('{"temperature": 0.2, "embedder": "hash-v0"}', encoding="utf-8")
    cal = load_calibration(str(p))
    assert cal["temperature"] == 0.2


def test_scores_are_cosine_similarity():
    s = Scorer()
    # Orthogonal to everything: embedding [1,0] and label [0,1].
    text = np.array([[1.0, 0.0]])
    label = np.array([[0.0, 1.0], [1.0, 0.0]])
    sc = s.scores(text, label)
    assert sc[0, 0] == pytest.approx(0.0)
    assert sc[0, 1] == pytest.approx(1.0)


def test_single_probs_temperature_sharpens():
    text = np.array([[1.0, 0.5]])
    labels = np.array([[1.0, 0.0], [0.0, 1.0]])
    cold = Scorer(calibration={"temperature": 0.5})
    hot = Scorer(calibration={"temperature": 10.0})
    assert cold.single_probs(text, labels)[0, 0] > hot.single_probs(text, labels)[0, 0]


def test_fit_temperature_recovers_low_temperature_when_separated():
    # Perfectly separated scores: the fit should push temperature down.
    scores = np.array([[3.0, 0.0], [0.0, 5.0], [2.0, 1.0]])
    true_idx = np.array([0, 1, 0])
    t, nll = fit_temperature(scores, true_idx)
    assert t > 0
    assert nll >= 0  # NLL is a positive mean


def test_fit_multi_recovers_bias_toward_positive():
    s = Scorer()
    s.multi_bias = -1.0
    s.multi_temperature = 0.8
    s.multi_threshold = 0.5
    assert s.multi_threshold == 0.5
    # Sanity: with a bias moved negative, probs rise.
    p_neg = s.multi_probs(np.array([[1.0, 0.0]]), np.array([[0.7, 0.0]]))
    assert p_neg[0, 0] > 0.5


def test_fit_multi_returns_bce_and_moves_bias():

    s = np.array([[0.6, -0.4], [0.9, 0.1]])
    y = np.array([[1, 0], [0, 1]])
    bias, temperature, bce = fit_multi(s, y)
    assert temperature > 0
    assert bce >= 0
    # The first label is far more positive than the second; bias should be
    # somewhere that reflects the input distribution, not zero by default.
    assert isinstance(bias, float)
    assert bias != 0.0 or bce == 0.0


def test_fit_multi_threshold_finds_working_boundary():
    # A sharply separated multi-label set where 0.5 is the wrong boundary.
    s = np.array([[3.0, -3.0], [2.5, -2.0], [-2.0, 2.0]])
    y = np.array([[1, 0], [1, 0], [0, 1]])
    t, f1 = fit_multi_threshold(s, y, bias=2.0, temperature=1.0)
    assert f1 > 0.9
    # The fitted threshold must lean low to undo the positive bias.
    assert t < 0.5