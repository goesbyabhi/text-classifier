"""
Minimal MCP server for text-classifier: one tool, `classify_texts`.

Speaks MCP over stdio (JSON-RPC 2.0, one message per line). The tool posts to
a text-classifier endpoint and returns the results as text content, so any MCP
host (Claude desktop, coding agents) can call classify with the same inputs
and labels you would send over HTTP. Stdlib only — no `mcp` package needed:

    python mcp/classify_server.py --endpoint https://text-classifier.example.workers.dev

Wire it up in `.mcp.json`:

    {"mcpServers": {"text-classifier": {"command": "python", "args": ["mcp/classify_server.py"]}}}

The whole protocol lives in `handle_message`, which is pure (no network in the
handler itself; urlopen is injected) so pytest can exercise it.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

DEFAULT_ENDPOINT = "https://text-classifier.abxisxekpanda.workers.dev"
PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "classify_texts",
        "description": (
            "Zero-shot classify a list of texts into the given labels. "
            "Returns one predicted label per text with calibrated confidence "
            "scores for every label, or (multi=True) every label above the "
            "calibrated threshold."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "inputs": {"type": "array", "items": {"type": "string"}, "description": "texts to classify"},
                "labels": {"type": "array", "items": {"type": "string"}, "minItems": 2, "description": "candidate labels"},
                "multi": {"type": "boolean", "default": False, "description": "multi-label mode"},
                "tier": {"type": "string", "enum": ["fast", "smart"], "default": "fast"},
                "max_labels": {"type": "integer", "description": "cap multi-label picks"},
            },
            "required": ["inputs", "labels"],
        },
    }
]


def _rpc(id, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    return msg


def _server_info():
    return {
        "name": "text-classifier",
        "description": "Zero-shot text classification over plain HTTP.",
        "version": "0.1.0",
    }


def call_classify(args, endpoint, urlopen=urllib.request.urlopen):
    """POST to /v1/classify. Returns (ok, payload_or_error)."""
    body = {
        "inputs": args.get("inputs", []),
        "labels": args.get("labels", []),
        "multi": bool(args.get("multi", False)),
        "tier": args.get("tier", "fast") if args.get("tier") in ("fast", "smart") else "fast",
    }
    if isinstance(args.get("max_labels"), int):
        body["max_labels"] = args["max_labels"]
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/classify",
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", "user-agent": "classify-mcp/0.1.0"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as res:
            reply = json.loads(res.read().decode("utf-8"))
        return True, reply
    except Exception as e:  # noqa: BLE001 - surface any upstream failure as an MCP error
        return False, str(e)


def handle_message(msg, endpoint=DEFAULT_ENDPOINT, urlopen=urllib.request.urlopen):
    """Handle one JSON-RPC message; returns the response dict or None for notifications."""
    if not isinstance(msg, dict):
        return _rpc(msg.get("id") if isinstance(msg, dict) else None, error={"code": -32600, "message": "Invalid Request"})
    method = msg.get("method")
    rid = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return _rpc(
            rid,
            result={
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": _server_info(),
            },
        )
    if method == "notifications/initialized":
        return None
    if method in ("tools/list", "mcp/tools/list"):
        return _rpc(rid, result={"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        if name != "classify_texts":
            return _rpc(rid, error={"code": -32602, "message": f"Unknown tool: {name}"})
        args = params.get("arguments") or {}
        ok, payload = call_classify(args, endpoint, urlopen=urlopen)
        if not ok:
            return _rpc(rid, error={"code": -32000, "message": f"classify failed: {payload}"})
        text = json.dumps(payload.get("results", payload), indent=2) if isinstance(payload, dict) else str(payload)
        return _rpc(rid, result={"content": [{"type": "text", "text": text}]})
    if method == "ping":
        return _rpc(rid, result={})
    return _rpc(rid, error={"code": -32601, "message": f"Method not found: {method}"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="base URL of the deployed Worker")
    args = ap.parse_args()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write("error: bad JSON-RPC line\n")
            sys.stderr.flush()
            continue
        response = handle_message(msg, endpoint=args.endpoint)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()