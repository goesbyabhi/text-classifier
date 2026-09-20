"""
Multi-label tagging: read a changelog and tag every entry with every label
that applies. In one request. The reference demo ran this over the Rust
changelog (language, compiler, library, tooling, platform) — an entry can be
both a library note and a language change, which is exactly the case where a
forced single label would hide noise.

    uv run python examples/changelog_tags.py            # offline
    uv run python examples/changelog_tags.py --endpoint http://127.0.0.1:8787
"""

from __future__ import annotations

import argparse

from lib import classifier_for, headline

LABELS = ["language", "compiler", "library", "tooling", "platform"]

CHANGELOG = [
    "stabilize `foo` in the standard library; the feature gate is now removed",
    "borrow checker: fix a false positive when matching on `&mut` bindings",
    "cargo now builds with the new resolver by default",
    "add `Unicode` data tables for the `char` API",
    "rustdoc: render hanging indents in doc comment lists",
    "the std simulator backend is removed from the nightly toolchain",
    "`impl Trait` in type aliases lands, without the older limitations",
    "LLVM upgraded to 19.1; codegen should be faster everywhere",
    "regex crate: switch to the smaller, faster memchr backend",
    "miri gains support for the new inline assembly syntax",
    "CLI: `cargo fix` can now autofix broken internal doc links",
    "the WASM target is promoted to tier-2 per the platform support policy",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default=None, help="base URL of a deployed Worker (default: offline)")
    args = ap.parse_args()

    clf = classifier_for(args.endpoint)
    print(headline(clf, "changelog tagging"))
    print(f"tags: {', '.join(LABELS)}\n")

    results = clf.classify(CHANGELOG, LABELS, multi=True)
    for i, r in enumerate(results):
        tags = ", ".join(r.get("labels") or []) or "(none)"
        print(f"  {tags:<34} {CHANGELOG[i][:54]}")


if __name__ == "__main__":
    main()