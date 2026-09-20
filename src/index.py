"""
The Worker entrypoint: routing, validation, scoring, rate limiting, analytics.

Top-level module code runs once at deploy time (the Worker snapshot), so the
calibration-scorer and the model list are built here, not per request. The
runtime SDK (`workers.*`) and this module are NOT valid outside the Worker;
all the math and parsing live in plain modules (classifier, embeddings,
query, limiter, unintelligible) that pytest exercises directly.

Request text is never logged or stored. Analytics rows carry tier, label
fingerprint, country, status, timing and counts only.
"""

from __future__ import annotations

import json
import time
from urllib.parse import parse_qs, unquote, urlparse

from workers import WorkerEntrypoint, Response

from benchmark import get_benchmark
from classifier import Scorer, load_calibration
from docs import DOCS
from embeddings import WorkersAIEmbedder
from home import page as home_page
from limiter import RateLimiter  # noqa: F401  (see wrangler class_name)
from query import USAGE, has_classify_query, read_get, suggest
from skill import index as skill_index
from skill import SKILL
from unintelligible import UNSCORED_REASON, is_unintelligible

API_VERSION = "v1"
MAX_INPUTS = 1000
MAX_LABELS = 100
MAX_CHARS = 32_000

CORS = {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "Content-Type, Authorization, Accept, Idempotency-Key",
    "access-control-expose-headers": "RateLimit-Limit, RateLimit-Remaining, RateLimit-Policy, Retry-After, x-api-version, Idempotency-Key",
}

# Both tiers serve the same frozen-embedding scorer today; the tier stays in
# the API surface for compatibility and to hang per-IP limits on.
TIERS = {
    "fast": {"rpm": 3000, "daily": 20_000},
    "smart": {"rpm": 200, "daily": 2_000},
}


def _to_py(proxy):
    if hasattr(proxy, "to_py"):
        return proxy.to_py()
    return proxy


def _classifier_id(labels):
    key = "|".join(sorted(l.lower().strip() for l in labels))
    return key[:120]


def _agent_family(ua: str) -> str:
    s = ua.lower().strip()
    if not s:
        return "none"
    if s.startswith("classify-cli/"):
        return "classify-cli"
    if s.startswith("curl/"):
        return "curl"
    if "python-requests" in s or "httpx" in s or "aiohttp" in s or s.startswith("python-urllib"):
        return "python"
    if "go-http-client" in s:
        return "go"
    if "bot" in s or "crawler" in s or "spider" in s:
        return "bot"
    if "mozilla" in s or "safari" in s:
        return "browser"
    return "other"


class Default(WorkerEntrypoint):
    """The Worker entrypoint. Also the home of the analytics helper."""

    async def fetch(self, request):
        url = urlparse(str(request.url))
        path = unquote(url.path).lstrip("/")
        ip = str(request.headers.get("cf-connecting-ip") or "anon")
        cf = _to_py(getattr(request, "cf", None))
        country = (cf or {}).get("country") or "??"
        origin = f"{url.scheme}://{url.netloc}"
        agent = _agent_family(str(request.headers.get("user-agent") or ""))

        if request.method == "OPTIONS":
            return Response("", status=204, headers=CORS)

        if request.method == "HEAD":
            headers = dict(CORS)
            headers["content-length"] = "0"
            return Response("", status=200, headers=headers)

        accept = str(request.headers.get("accept") or "")
        want_html = "text/html" in accept and "application/json" not in accept
        want_markdown = "text/markdown" in accept
        accept_json = "application/json" in accept

        if request.method == "GET" and path in ("health", "v1/health"):
            body = {"ok": True, "service": "text-classifier", "version": API_VERSION}
            return self._json(body, 200, {"cache-control": "no-store"})

        if request.method == "GET" and not path:
            params = parse_qs(urlparse(str(request.url)).query)
            if has_classify_query(params):
                return await self._classify(request, path, origin, ip, country, agent)
            wants_json = accept_json and "text/html" not in accept
            html_headers = {"cache-control": "public, max-age=60", "vary": "accept"}
            if want_html and not wants_json:
                return self._html(home_page(origin), 200, html_headers)
            return self._docs(origin, wants_json, want_markdown)

        if request.method == "GET" and path == "openapi.json":
            return self._json(self._openapi(origin), 200, {"cache-control": "public, max-age=3600"})

        # ---- discovery surfaces --------------------------------------------
        if request.method == "GET" and path == "benchmark":
            return self._text(get_benchmark(), 200, {"cache-control": "public, max-age=3600"})

        if request.method == "GET" and path == "skill.md":
            return self._text(SKILL, 200, {"content-type": "text/markdown; charset=utf-8", "cache-control": "public, max-age=3600"})

        if request.method == "GET" and path == ".well-known/agent-skills/index.json":
            return self._json(skill_index(), 200, {"cache-control": "public, max-age=3600"})

        # ---- classification ------------------------------------------------
        if request.method in ("GET", "POST"):
            return await self._classify(request, path, origin, ip, country, agent)
        return self._not_found(origin, accept)

    # ---- response helpers -------------------------------------------------

    def _json(self, body, status=200, extra=None):
        headers = {"content-type": "application/json; charset=utf-8", **CORS}
        if extra:
            headers.update(extra)
        return Response(json.dumps(body, indent=2), status=status, headers=headers)

    def _text(self, body, status=200, extra=None):
        headers = {"content-type": "text/plain; charset=utf-8", **CORS}
        if extra:
            headers.update(extra)
        return Response(body, status=status, headers=headers)

    def _html(self, body, status=200, extra=None):
        headers = {"content-type": "text/html; charset=utf-8", **CORS}
        if extra:
            headers.update(extra)
        return Response(body, status=status, headers=headers)

    def _docs(self, origin, wants_json, wants_markdown):
        link = {
            "link": (
                f'<{origin}/openapi.json>; rel="service-desc"; type="application/openapi+json", '
                f'<{origin}/health>; rel="service-health", '
                f'<{origin}/.well-known/agent-skills/index.json>; rel="agent-skills"; type="application/json"'
            )
        }
        if wants_json:
            return self._json(self._agent_view(origin), 200, {**link, "cache-control": "public, max-age=3600"})
        if wants_markdown:
            return self._text(DOCS, 200, {**link, "cache-control": "public, max-age=3600", "vary": "accept"})
        return self._text(DOCS, 200, {**link, "cache-control": "public, max-age=3600", "vary": "accept"})

    def _agent_view(self, origin):
        return {
            "name": "text-classifier",
            "description": "Zero-shot text classification over plain HTTP. No API key, no account.",
            "version": API_VERSION,
            "authentication": {"required": False},
            "api": {
                "classify": {"method": "POST", "url": f"{origin}/v1/classify", "body": {"inputs": ["..."], "labels": ["a", "b"], "tier": "fast|smart", "multi": False}},
                "classify_one": {"method": "GET", "url": f"{origin}/{{labels}}/{{text}}", "query_form": f"{origin}/?labels={{a,b}}&text={{text}}"},
                "openapi": f"{origin}/openapi.json",
            },
            "limits": {"fast": "3,000 classifications/min, 20,000/day per IP", "smart": "200/min, 2,000/day per IP"},
            "docs": f"{origin}/",
            "benchmark": f"{origin}/benchmark",
            "skill": {"md": f"{origin}/skill.md", "discovery": f"{origin}/.well-known/agent-skills/index.json"},
        }

    def _openapi(self, origin):
        return {
            "openapi": "3.1.0",
            "info": {"title": "text-classifier", "version": API_VERSION},
            "servers": [{"url": origin}],
            "paths": {
                "/v1/classify": {
                    "post": {
                        "operationId": "classify",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["inputs", "labels"],
                                        "properties": {
                                            "inputs": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": MAX_INPUTS},
                                            "labels": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": MAX_LABELS},
                                            "tier": {"type": "string", "enum": ["fast", "smart"], "default": "fast"},
                                            "multi": {"type": "boolean", "default": False},
                                            "max_labels": {"type": "integer"},
                                        },
                                    }
                                }
                            }
                        },
                        "responses": {"200": {"description": "classified"}},
                    }
                },
                "/health": {"get": {"operationId": "health", "responses": {"200": {"description": "ok"}}}},
                "/benchmark": {"get": {"operationId": "benchmark", "responses": {"200": {"description": "plain-text measured benchmark"}}}},
                "/skill.md": {"get": {"operationId": "skill", "responses": {"200": {"description": "agent skill artifact (markdown)"}}}},
                "/.well-known/agent-skills/index.json": {"get": {"operationId": "agentSkillsIndex", "responses": {"200": {"description": "RFC-8615 agent-skills discovery index"}}}},
            },
        }

    def _not_found(self, origin, accept):
        msg = "Not found. The API is POST /v1/classify, GET /{labels}/{text} or GET /?labels=a,b&text=..."
        if "application/json" in accept:
            return self._json({"error": msg, "code": "not_found", "see": {"docs": f"{origin}/"}}, 404)
        return self._text(f"# Not found\n\n{msg}\n\nStart at {origin}/\n", 404)

    # ---- classification pipeline -----------------------------------------

    def _embed(self):
        """Build (once per isolate) the embedder and scorer from the binding."""
        if getattr(self, "_embedder", None) is None:
            self._embedder = WorkersAIEmbedder(self.env.AI)
            self._scorer = Scorer(load_calibration(None, embedder=self._embedder.name))
        return self._embedder, self._scorer

    async def _classify(self, request, path, origin, ip, country, agent):
        embedder, scorer = self._embed()
        tier = "fast"
        inputs = []
        labels = []
        instructions = None
        multi = False
        max_labels = None
        want_json = request.method == "POST"
        get_req = None

        if request.method == "POST":
            try:
                body = _to_py(await request.json())
            except Exception:
                self._record(ip, country, tier, [], status=400, agent=agent, reason="bad_json", ms=0, n=0)
                return self._json({"error": "Body must be JSON. See /"}, 400)
            inputs = (
                body.get("inputs", [])
                if isinstance(body.get("inputs"), list)
                else [body["input"]] if isinstance(body.get("input"), str) else []
            )
            labels = body.get("labels", []) if isinstance(body.get("labels"), list) else []
            if body.get("tier") == "smart":
                tier = "smart"
            if isinstance(body.get("instructions"), str):
                instructions = body["instructions"]
            if body.get("multi") is True or isinstance(body.get("max_labels"), (int, float)):
                multi = True
                max_labels = body.get("max_labels")
                if isinstance(max_labels, (int, float)) and max_labels > 0 and not isinstance(max_labels, bool):
                    max_labels = int(max_labels)
                else:
                    max_labels = None
        else:
            params = parse_qs(urlparse(str(request.url)).query)
            get_req = read_get(path, params)
            if get_req.nothing:
                return self._not_found(origin, str(request.headers.get("accept") or ""))
            labels = get_req.labels
            inputs = [get_req.text]
            tier = get_req.tier
            instructions = get_req.instructions
            multi = get_req.multi
            max_labels = get_req.multi_max
            want_json = get_req.verbose

        rate_limit = TIERS[tier]["rpm"]

        def yield_headers(remaining):
            limit = TIERS[tier]["rpm"]
            headers = {
                "x-api-version": API_VERSION,
                "ratelimit-limit": str(limit),
                "ratelimit-policy": f"{limit};w=60, {TIERS[tier]['daily']};w=86400",
                "x-ratelimit-limit": f"{limit}/min",
            }
            if remaining >= 0:
                headers["ratelimit-remaining"] = str(remaining)
                headers["x-ratelimit-remaining"] = str(remaining)
            return headers

        def fail(msg, status, reason, extra=None, ms=0):
            self._record(ip, country, tier, labels, status=status, agent=agent, reason=reason, ms=ms, n=len(inputs))
            headers = yield_headers(-1)
            if extra:
                headers.update(extra)
            if want_json:
                hint = {"usage": USAGE, "try": suggest(origin, get_req)} if status == 400 and get_req else None
                body = {"error": msg, "code": reason}
                if hint:
                    body.update(hint)
                return self._json(body, status, headers)
            return self._text(f"error: {msg}\n", status, headers)

        # ---- validation ----------------------------------------------------
        if not inputs or not inputs[0]:
            return fail("Provide text to classify. See /", 400, "no_input")
        non_string = [i for i in inputs if not isinstance(i, str)]
        if non_string:
            return fail("Inputs must be strings", 400, "empty_input")
        if len(inputs) > MAX_INPUTS:
            return fail(f"Maximum {MAX_INPUTS} inputs per request", 400, "too_many_inputs")
        if len(labels) < 2:
            return fail(
                "Provide at least 2 labels, separated by commas" if get_req else "Provide at least 2 labels",
                400,
                "too_few_labels",
            )
        if len(labels) > MAX_LABELS:
            return fail(f"Maximum {MAX_LABELS} labels", 400, "too_many_labels")
        if any(not isinstance(l, str) or not l.strip() for l in labels):
            return fail("Labels must be non-empty strings", 400, "empty_label")
        if len(set(labels)) != len(labels):
            return fail("Labels must be distinct", 400, "duplicate_labels")
        if any(not i.strip() for i in inputs):
            return fail("Inputs must be non-empty strings", 400, "empty_input")
        if any(len(i) > MAX_CHARS for i in inputs):
            return fail(f"Each input must be under {MAX_CHARS} characters", 400, "input_too_long")

        # ---- rate limiting -------------------------------------------------
        gate = await self._limited(tier, ip, len(inputs))
        if gate["limited"]:
            per_day = gate.get("scope") == "day"
            return fail(
                f"Daily limit reached: {TIERS[tier]['daily']} {tier} classifications per IP per day"
                if per_day
                else f"Rate limit: {rate_limit} {tier} classifications/minute per IP",
                429,
                "rate_limit_day" if per_day else "rate_limit_minute",
                {
                    "retry-after": str(gate.get("reset_in") or 60),
                    "ratelimit-remaining": "0",
                    "x-ratelimit-remaining": "0",
                },
            )

        # ---- score ---------------------------------------------------------
        started = time.time()
        try:
            label_embs = await embedder.embed_texts(labels)
            text_embs = await embedder.embed_texts(inputs)
        except Exception as e:  # noqa: BLE001
            return fail(f"upstream: {e}", 502, "upstream_other", ms=int((time.time() - started) * 1000))

        results = []
        if multi:
            probs = scorer.multi_probs(text_embs, label_embs)
            for i, row in enumerate(probs):
                order = sorted(range(len(labels)), key=lambda j: -row[j])
                picked = [labels[j] for j in order if row[j] >= scorer.multi_threshold]
                if max_labels:
                    picked = picked[:max_labels]
                unreadable = is_unintelligible(inputs[i])
                results.append(
                    {
                        "labels": picked,
                        "scores": None if unreadable else dict(zip(labels, [round(float(v), 4) for v in row])),
                        "unscored": UNSCORED_REASON if unreadable else None,
                    }
                )
        else:
            probs = scorer.single_probs(text_embs, label_embs)
            for i, row in enumerate(probs):
                argmax = int(row.argmax())
                unreadable = is_unintelligible(inputs[i])
                results.append(
                    {
                        "label": labels[argmax],
                        "confidence": None if unreadable else round(float(row[argmax]), 4),
                        "scores": None if unreadable else dict(zip(labels, [round(float(v), 4) for v in row])),
                        "unscored": UNSCORED_REASON if unreadable else None,
                    }
                )

        ms = int((time.time() - started) * 1000)
        for r in results:
            r["ms"] = ms
            r["model"] = embedder.name

        self._record(ip, country, tier, labels, status=200, agent=agent, reason="", ms=ms, n=len(results))
        headers = yield_headers(gate.get("remaining", -1))

        if not want_json:
            body = "\n".join(
                "\n".join(r["labels"] or []) if multi else r["label"] for r in results
            )
            return self._text(body + "\n", 200, headers)

        if request.method == "GET":
            return self._json({**results[0], "tier": tier}, 200, headers)

        models_used = sorted({r["model"] for r in results if r["model"]})
        return self._json(
            {
                "tier": tier,
                "model": models_used[0] if len(models_used) == 1 else ("mixed" if models_used else ""),
                "modelsUsed": models_used,
                "results": results,
                "usage": {"classifications": len(results), "ms": ms},
            },
            200,
            headers,
        )

    # ---- rate limiting + analytics ---------------------------------------

    async def _limited(self, tier, ip, cost):
        try:
            limiter_id = self.env.LIMITER.idFromName(f"{tier}:{ip}")
            stub = self.env.LIMITER.get(limiter_id)
            res = await stub.check(TIERS[tier]["rpm"], TIERS[tier]["daily"], cost)
            res = _to_py(res)
            return {
                "limited": bool(res.get("limited", False)),
                "remaining": int(res.get("remaining", -1)),
                "scope": res.get("scope"),
                "reset_in": res.get("reset_in"),
            }
        except Exception:  # noqa: BLE001
            # Never fail closed because the limiter had a bad moment.
            return {"limited": False, "remaining": -1, "scope": None, "reset_in": None}

    def _record(self, ip, country, tier, labels, status, agent, reason, ms, n):
        try:
            self.env.AE.writeDataPoint(
                {
                    "blobs": [tier, _classifier_id(labels) or "-", country, str(status), agent, reason],
                    "doubles": [n, ms],
                    "indexes": [ip[:32]],
                }
            )
        except Exception:  # noqa: BLE001 - analytics must never break a request
            pass
