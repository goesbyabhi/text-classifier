"""
The GET surface: two spellings of the same request.

The path form is the one people type: /spam,not+spam/Win+a+free+iPhone. The
query form is the one URL-state libraries emit: /?labels=spam,not+spam&text=...
Ported from classifier.dev's src/query.ts, including the label/text aliases an
agent guesses before reading the docs.
"""

from __future__ import annotations

from urllib.parse import parse_qs

ALIASES = {
    "text": ["input", "content", "message", "body", "sentence", "query", "q"],
    "labels": ["label", "classes", "categories", "options"],
}

USAGE = "GET /{labels}/{text}  or  GET /?labels={a,b}&text={text}"


def _plus(s: str) -> str:
    return s.replace("+", " ").strip()


def _split_labels(s: str) -> list[str]:
    return [x.replace("+", " ").strip() for x in s.split(",") if x.replace("+", " ").strip()]


def has_classify_query(params: dict) -> bool:
    """True when the query string carries the request itself, not just options."""
    for key in ["text", "labels", *ALIASES["text"], *ALIASES["labels"]]:
        if key in params:
            return True
    return False


class GetRequest:
    def __init__(
        self,
        labels,
        text,
        tier,
        instructions,
        verbose,
        form,
        nothing,
        multi_max=None,
        multi=False,
    ):
        self.labels = labels
        self.text = text
        self.tier = tier
        self.instructions = instructions
        self.verbose = verbose
        self.form = form
        self.nothing = nothing
        self.multi_max = multi_max
        self.multi = multi or multi_max is not None


def read_get(path: str, params: dict) -> GetRequest:
    """Read a GET classification request from the decoded path and query
    params (already parsed via parse_qs). The query string wins where both say
    something; the path fills whatever it left out."""
    # Aliases: if the canonical key is absent, take the first alias that hit.
    for canonical, names in ALIASES.items():
        if canonical in params:
            continue
        for name in names:
            if name in params:
                params[canonical] = params[name]
                break

    slash = path.find("/")
    path_labels = _split_labels(path[:slash] if slash > 0 else path)
    path_text = _plus(path[slash + 1 :]) if slash > 0 else ""
    from_query = has_classify_query(params)

    # The query form carries labels as one comma-joined value (?labels=a,b);
    # split it the same way the path form is split.
    labels = _split_labels(",".join((params.get("labels") or [""])))
    if not labels:
        labels = path_labels

    text = params.get("text", [None])[0]
    if text is not None:
        text = text.strip()
    else:
        text = path_text

    tier = "fast"
    if "tier" in params and params["tier"][0] in ("fast", "smart"):
        tier = params["tier"][0]

    instructions = params.get("instructions", [None])[0]

    verbose = params.get("verbose", ["0"])[0].lower() in ("1", "true", "yes", "on")
    multi_raw = params.get("multi", ["0"])[0].lower() in ("1", "true", "yes", "on")
    max_raw = params.get("max_labels", [None])[0]
    multi_max = None
    if max_raw is not None and max_raw.isdigit():
        multi_max = int(max_raw) or None
    if multi_max is None and multi_raw:
        multi_max = None  # flag form of multi, unbounded
    multi = multi_raw or multi_max is not None

    nothing = not from_query and slash <= 0 and "," not in path

    return GetRequest(
        labels=labels,
        text=text,
        tier=tier,
        instructions=instructions,
        verbose=verbose,
        form="query" if from_query else "path",
        nothing=nothing,
        multi_max=multi_max,
        multi=multi,
    )


def suggest(origin: str, req: GetRequest) -> str:
    """A URL that would have worked, built from what the caller sent."""
    labels = list(dict.fromkeys([l for l in req.labels if l.strip()]))
    if not labels:
        labels = ["spam", "not spam"]
    elif len(labels) == 1:
        labels = [labels[0], f"not {labels[0]}"]
    text = req.text if 0 < len(req.text) <= 200 else "Win a free iPhone"
    if req.form == "query":
        qlabels = ",".join(labels)
        return f"{origin}/?labels={qlabels}&text={'+'.join(text.split())}"
    path_labels = ",".join(labels).replace(" ", "+")
    path_text = "+".join(text.split())
    return f"{origin}/{path_labels}/{path_text}"