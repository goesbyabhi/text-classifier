"""
classify: the one-shot CLI for text-classifier.

Reads lines from stdin (or files) and POSTs them to a deployed
text-classifier instance, printing one predicted label per line so the
output stays greppable in pipelines:

    classify spam,not+spam < feedback.txt
    classify --multi bug,feature,praise < tickets.txt
    cat notes.txt | classify --endpoint https://tc.example.com urgent,quiet

stdlib only; no install-time dependency beyond the examples already in
pyproject.toml. Runs anywhere, not just inside the Worker.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

DEFAULT_ENDPOINT = "https://text-classifier.<your-subdomain>.workers.dev"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", help="comma-separated labels")
    ap.add_argument("files", nargs="*", help="files to read (default: stdin)")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="base URL of the deployed Worker")
    ap.add_argument("--multi", action="store_true", help="multi-label classification")
    ap.add_argument("--tier", default="fast", choices=["fast", "smart"])
    ap.add_argument("--verbose", action="store_true", help="print the JSON response")
    args = ap.parse_args()

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if len(labels) < 2:
        print("error: provide at least 2 comma-separated labels", file=sys.stderr)
        raise SystemExit(1)

    if args.files:
        inputs = []
        for f in args.files:
            with open(f, encoding="utf-8") as fh:
                inputs.extend(line.rstrip("\n") for line in fh if line.strip())
    else:
        inputs = [line.rstrip("\n") for line in sys.stdin if line.strip()]

    if not inputs:
        print("error: no input", file=sys.stderr)
        raise SystemExit(1)

    body = {"inputs": inputs, "labels": labels, "tier": args.tier, "multi": args.multi}
    req = urllib.request.Request(
        f"{args.endpoint}/v1/classify",
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", "user-agent": "classify-cli/0.1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            reply = json.loads(res.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"error: could not reach {args.endpoint}: {e}", file=sys.stderr)
        raise SystemExit(1)

    if args.verbose:
        print(json.dumps(reply, indent=2))
        return

    for r in reply.get("results", []):
        print("\n".join(r.get("labels") or []) if args.multi else r.get("label", ""))


if __name__ == "__main__":
    main()