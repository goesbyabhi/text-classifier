"""
Triage a support inbox into billing / bug / feature / other, and flag the ones
a human should double-check.

Each ticket is classified in one batch; the send-to-who logic is the honest
part. Tickets whose top confidence clears the bar route themselves (billing to
finance, bug to eng); anything below it is flagged rather than trusted by an
automation, because the number is calibrated and 0.6 means 60%, not a
conviction.

Data is a small, realistic inbox embedded here so the example runs anywhere
(the reference demo invents the emails the same way).

    uv run python examples/support_triage.py            # offline
    uv run python examples/support_triage.py --endpoint http://127.0.0.1:8787
"""

from __future__ import annotations

import argparse

from lib import classifier_for, confidence_of, headline, pct

LABELS = ["billing", "bug", "feature", "other"]
BAR = 0.6  # below this, a human decides

INBOX = [
    "hi, my invoice shows double the amount this month and I never upgraded. please fix.",
    "the app crashes the moment I try to upload a photo from the gallery.",
    "would be great if you supported exporting everything as csv, we need it for our tax records.",
    "I want to cancel my pro plan but I can't find where to do it. also refund for march?",
    "when I paste a long link it breaks the formatting and the button disappears.",
    "is there any chance of an offline mode? sometimes I have no signal on the train.",
    "just wanted to say the new dashboard is really nice, thanks for making it fast.",
    "my account balance is wrong, it keeps showing 45 when I deposited 55.",
    "the login page hangs after entering my password, only on safari on ios.",
    "can you split payments across two cards? my company gives me a card for that.",
    "help, double charge for the annual plan, want the difference back.",
    "error code 503 when syncing, started this morning, I've tried restarting everything.",
    "a dark theme toggle would save my eyes, we work late shifts.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default=None, help="base URL of a deployed Worker (default: offline)")
    ap.add_argument("--bar", type=float, default=BAR, help="confidence below which a human decides")
    args = ap.parse_args()

    clf = classifier_for(args.endpoint)
    print(headline(clf, "support triage"))
    print(f"routing rule: confidence >= {args.bar} routes itself, below it a human decides\n")

    results = clf.classify(INBOX, LABELS)
    routes = {"billing": "finance", "bug": "eng", "feature": "product", "other": "human review"}
    for i, r in enumerate(results):
        conf = confidence_of(r)
        dest = routes.get(r["label"], "human review")
        human = "  <-- HUMAN DOUBLE-CHECK" if (conf or 0) < args.bar else ""
        print(f"#{i + 1:>2} {r['label']:<8} {pct(conf)}  -> {dest:<12} {INBOX[i][:58]}{human}")

    flagged = sum(1 for r in results if (confidence_of(r) or 0) < args.bar)
    print(f"\n{len(results)} tickets, {flagged} flagged for a human")


if __name__ == "__main__":
    main()