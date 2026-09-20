"""
Classify the current Hacker News front page the way people use classifier.dev.

Fetches the live HN front page (Algolia's public API, no key), buckets every
story title into ai / security / business / hardware / other in ONE request,
then prints the counts per bucket and the three stories the model is least
confident about — the ones worth reading before believing the bucket.

    uv run python examples/hn_headlines.py            # offline hash stand-in
    uv run python examples/hn_headlines.py --endpoint http://127.0.0.1:8787

A deployed Worker answers the whole batch in one POST; that is the point.
"""

from __future__ import annotations

import argparse

from lib import classifier_for, fetch_json, headline, pct

LABELS = ["ai", "security", "business", "hardware", "other"]

FRONT_PAGE_URL = "https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage=30"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default=None, help="base URL of a deployed Worker (default: offline)")
    args = ap.parse_args()

    clf = classifier_for(args.endpoint)
    print(headline(clf, "hn front page"))

    data = fetch_json(FRONT_PAGE_URL)
    titles = [h.get("title") for h in data.get("hits", []) if h.get("title")]
    if not titles:
        print("no stories came back from the front page")
        return

    results = clf.classify(titles, LABELS)
    counts = {lab: 0 for lab in LABELS}
    for r in results:
        counts[r["label"]] += 1

    print(f"classified {len(titles)} stories -> {counts}\n")

    ranked = sorted(
        ((confidence_of(r), titles[i]) for i, r in enumerate(results)),
        key=lambda t: t[0],
    )
    print("least confident (flag for a human):")
    for conf, title in ranked[:3]:
        print(f"  {pct(conf)}  {title}")


def confidence_of(r) -> float:
    return r.get("confidence")


if __name__ == "__main__":
    main()