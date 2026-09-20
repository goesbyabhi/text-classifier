import json

from classify_server import TOOLS, call_classify, handle_message


def _msg(method, params=None, rid=1):
    msg = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(payload):
    def opener(request, timeout=60):
        return _FakeResp(payload)

    return opener


def test_initialize_handshake():
    out = handle_message(_msg("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {}}))
    assert out["result"]["serverInfo"]["name"] == "text-classifier"
    assert "tools" in out["result"]["capabilities"]


def test_initialized_notification_gets_no_reply():
    assert handle_message(_msg("notifications/initialized", rid=None)) is None


def test_tools_list_exposes_classify_texts():
    out = handle_message(_msg("tools/list"))
    assert out["result"]["tools"] == TOOLS
    assert TOOLS[0]["name"] == "classify_texts"
    assert "inputs" in TOOLS[0]["inputSchema"]["properties"]
    assert "labels" in TOOLS[0]["inputSchema"]["required"]


def test_ping():
    out = handle_message(_msg("ping"))
    assert out["result"] == {}


def test_unknown_tool_returns_error():
    out = handle_message(_msg("tools/call", {"name": "nope", "arguments": {}}))
    assert out["error"]["code"] == -32602


def test_unknown_method_returns_error():
    out = handle_message(_msg("bogus/method"))
    assert out["error"]["code"] == -32601


def test_valid_request_is_not_accepted():
    out = handle_message("not-a-dict")
    assert out["error"]["code"] == -32600


def test_call_classify_proxies_and_returns_results():
    payload = {
        "results": [{"label": "spam", "confidence": 0.9912, "scores": {"spam": 0.9912, "not spam": 0.0088}}],
        "model": "@cf/baai/bge-small-en-v1.5",
    }
    urlopen = _fake_urlopen(payload)
    out = handle_message(
        _msg("tools/call", {"name": "classify_texts", "arguments": {"inputs": ["Win a free iPhone"], "labels": ["spam", "not spam"]}}),
        urlopen=urlopen,
    )
    assert out["result"]["content"][0]["type"] == "text"
    assert '"spam"' in out["result"]["content"][0]["text"]


def test_call_classify_forwards_multi_and_tier():
    seen = {}

    def urlopen(request, timeout=60):
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResp({"results": []})

    handle_message(
        _msg("tools/call", {"name": "classify_texts", "arguments": {"inputs": ["a"], "labels": ["x", "y"], "multi": True, "tier": "smart", "max_labels": 1}}),
        urlopen=urlopen,
    )
    assert seen["body"]["multi"] is True
    assert seen["body"]["tier"] == "smart"
    assert seen["body"]["max_labels"] == 1


def test_call_classify_upstream_failure_is_mcp_error():
    def boom(request, timeout=60):
        raise OSError("connection refused")

    out = handle_message(
        _msg("tools/call", {"name": "classify_texts", "arguments": {"inputs": ["a"], "labels": ["x", "y"]}}),
        urlopen=boom,
    )
    assert out["error"]["code"] == -32000
    assert "connection refused" in out["error"]["message"]


def test_call_classify_standalone_helper():
    ok, payload = call_classify(
        {"inputs": ["a"], "labels": ["x", "y"]},
        "https://example.invalid",
        urlopen=_fake_urlopen({"results": [{"label": "x", "confidence": 0.9}]}),
    )
    assert ok is True
    assert payload["results"][0]["label"] == "x"