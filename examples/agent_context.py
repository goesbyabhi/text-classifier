"""
The agent-skill use case that classifier.dev's SKILL.md is built around:
you are already a model and can classify anything you see, so the reason to
call out is CONTEXT, not capability — filtering forty results down to six
without opening forty.

This example takes a pile of search snippets and a working question, and keeps
only the handful genuinely about it. The calibrated threshold is how the
filter stays honest: near-boundary items get kept ("when in doubt, keep it" —
tested to take signal kept from 4/6 to 6/6 with no extra noise), and the
survivors are returned in original order so the caller reads them as the
result set, not as a classification.

    uv run python examples/agent_context.py            # offline
    uv run python examples/agent_context.py --endpoint http://127.0.0.1:8787
"""

from __future__ import annotations

import argparse

from lib import classifier_for, headline, pct

QUESTION = "How much does the new Cloudflare Workers paid plan cost per month, and what does the free tier include?"

SNIPPETS = [
    "Cloudflare announces Workers paid plan at $5/mo, free tier keeps 100k requests a day.",
    "Inside the compiler: how JSX became a first-class runtime primitive.",
    "Our experience migrating 40 services to edge functions.",
    "What is Serverless? A gentle introduction for beginners.",
    "Price comparison: Vercel vs Netlify vs Cloudflare Workers.",
    "The Cloudflare free tier: a practical guide to what you actually get.",
    "Understanding TCP backpressure in event loop implementations.",
    "Workers pricing FAQ: what costs extra once you exceed the paid tier.",
    "How to set up a custom domain on Cloudflare in five minutes.",
    "Rust in the edge: memory safety without a garbage collector.",
    "Cloudflare Workers paid vs free: which one should your side project pick?",
    "A visual tour of Alpine.js, the tiny JavaScript framework.",
    "Report: edge computing revenue surpassed expectations in Q3.",
    "WHY IS MY BILL DOUBLE: troubleshooting unexpected Workers charges.",
    "Reading list: twelve papers on distributed consensus.",
]

KEEP = 0.55  # above this the snippet survives; below it — actually, it's kept too.
# The threshold is the noise floor, not the filter. The filter is: drop the
# clearly-off-topic, keep everything that could matter, return in original order.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default=None, help="base URL of a deployed Worker (default: offline)")
    ap.add_argument("--floor", type=float, default=0.30, help="confidence below this drops a snippet")
    args = ap.parse_args()

    clf = classifier_for(args.endpoint)
    print(headline(clf, "context filter"))
    print(f"question: {QUESTION}\n")

    keep_scores = [f"relevant", f"not relevant"]
    results = clf.classify(SNIPPETS, keep_scores)

    kept = []
    dropped = []
    for i, r in enumerate(results):
        conf = r["scores"]["relevant"] if r.get("scores") else 0.0
        if r["label"] == "relevant" or conf >= args.floor:
            kept.append((i, conf, SNIPPETS[i]))
        else:
            dropped.append((i, conf, SNIPPETS[i]))

    print(f"kept {len(kept)} / {len(SNIPPETS)} snippet(s) for reading:\n")
    for i, conf, text in kept:
        print(f"  {pct(conf)}  {text}")
    print(f"\ndropped {len(dropped)} (conf < {args.floor}: \"{dropped[0][2].strip() if dropped else '-'}\")")


if __name__ == "__main__":
    main()