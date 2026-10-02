"""Polite, cached client for the Portuguese Wikipedia API.

Every response is cached on disk (gzip JSON keyed by the request parameters), so an interrupted
download resumes where it stopped and re-cleaning the corpus never touches the network again.
Network requests are spaced by ``delay_s``; HTTP 429/5xx responses wait for ``Retry-After`` or
an exponential backoff, whichever is longer (the API rate-limited the feasibility study
otherwise). Transient failures are retried and, after ``max_attempts``, raise instead of being
skipped. Cache files are written atomically, and ``offline=True`` turns a cache miss into an error.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gender_networks.artifacts import read_json_gz, write_json_gz

LOGGER = logging.getLogger(__name__)
API_URL = "https://pt.wikipedia.org/w/api.php"

# transport(url, headers, timeout) -> (status, headers, body)
Transport = Callable[[str, dict[str, str], float], tuple[int, dict[str, str], bytes]]


def urllib_transport(url: str, headers: dict[str, str], timeout: float) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read() or b""


class ApiError(RuntimeError):
    """The API answered with a permanent error (e.g. an invalid category); retrying won't help."""


class CacheMiss(RuntimeError):
    """Offline mode needed a response that is not in the cache."""


# Error codes that mean the request itself is wrong; every other code (ratelimited, readonly,
# internal_api_error_*, ...) is treated as transient and retried with backoff.
PERMANENT_ERRORS = {
    "invalidcategory", "invalidtitle", "badvalue", "baddatatype", "badinteger", "missingparam",
    "toomanyvalues", "unknown_action", "nosuchpageid", "nosuchrevid", "paramempty",
}  # fmt: skip
_RETRYABLE = (OSError, http.client.HTTPException, ValueError)  # ValueError: bad JSON or UTF-8


@dataclass(frozen=True)
class Page:
    """Latest revision of an article."""

    title: str
    pageid: int
    revid: int
    wikitext: str
    disambiguation: bool = False


class WikiClient:
    """Minimal MediaWiki API client with on-disk cache, spacing and retries.

    With ``offline=True`` a cache miss raises :class:`CacheMiss`, so a rebuild from the cache
    either reproduces the cached selection or fails loudly instead of fetching newer revisions.
    """

    def __init__(
        self,
        cache_dir: Path,
        user_agent: str,
        delay_s: float = 2.5,
        api_url: str = API_URL,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 12,
        offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.user_agent = user_agent
        self.delay_s = delay_s
        self.api_url = api_url
        self.transport = transport or urllib_transport
        self.sleep = sleep
        self.max_attempts = max_attempts
        self.offline = offline
        self.network_requests = 0
        self.cache_hits = 0
        self.skipped_categories: list[str] = []

    def _cache_path(self, params: dict[str, str]) -> Path:
        key = json.dumps(sorted(params.items()), ensure_ascii=False)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json.gz"

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        """Cached response, or None (a corrupt file from an interrupted write is deleted)."""

        if not path.exists():
            return None
        try:
            return read_json_gz(path)
        except (EOFError, OSError, ValueError) as error:
            LOGGER.warning("Cache corrompido %s (%s); será baixado de novo", path, error)
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, path: Path, data: dict[str, Any]) -> None:
        """Atomic write: a killed process never leaves a truncated cache file behind."""

        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        write_json_gz(temporary, data)
        os.replace(temporary, path)

    @staticmethod
    def _checked(data: dict[str, Any]) -> dict[str, Any]:
        error = data.get("error")
        if error:
            raise ApiError(f"API error {error.get('code')}: {error.get('info')}")
        return data

    def request(self, **params: str) -> dict[str, Any]:
        """GET the API with JSON output; cached responses never reach the network.

        Permanent API errors are cached too, so a rerun walks the category tree identically.
        """

        full = {"format": "json", "formatversion": "2", **{k: str(v) for k, v in params.items()}}
        path = self._cache_path(full)
        cached = self._read_cache(path)
        if cached is not None:
            self.cache_hits += 1
            return self._checked(cached)
        if self.offline:
            raise CacheMiss(f"Offline and not cached: {full}")
        url = self.api_url + "?" + urllib.parse.urlencode({**full, "maxlag": "5"})
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "identity"}
        for attempt in range(1, self.max_attempts + 1):
            self.sleep(self.delay_s)
            self.network_requests += 1
            backoff = min(600.0, 30.0 * 2 ** (attempt - 1))
            data: dict[str, Any] = {}
            try:
                status, response_headers, body = self.transport(url, headers, 60.0)
                if status == 200:
                    data = json.loads(body.decode("utf-8"))
            except _RETRYABLE as error:
                wait = min(120.0, 5.0 * attempt)
                LOGGER.warning("Erro de rede (%r); nova tentativa em %.0f s", error, wait)
                self.sleep(wait)
                continue
            if status == 429 or status >= 500:
                wait = max(_retry_after(response_headers), backoff)
                LOGGER.warning("HTTP %s; aguardando %.0f s (tentativa %d)", status, wait, attempt)
                self.sleep(wait)
                continue
            if status != 200:
                raise RuntimeError(f"HTTP {status} for {url}")
            code = str(data.get("error", {}).get("code", ""))
            if code == "maxlag":
                self.sleep(max(10.0, _retry_after(response_headers)))
                continue
            if code and code not in PERMANENT_ERRORS:
                LOGGER.warning("Erro transitório da API (%s); aguardando %.0f s", code, backoff)
                self.sleep(backoff)
                continue
            self._write_cache(path, data)
            return self._checked(data)
        raise RuntimeError(f"Giving up after {self.max_attempts} attempts: {url}")

    def category_members(
        self, category: str, skip_patterns: Iterable[str] = ()
    ) -> tuple[list[str], list[str]]:
        """Articles (namespace 0) and content subcategories (namespace 14) of a category."""

        skip = tuple(skip_patterns)
        pages: list[str] = []
        subcats: list[str] = []
        cont: dict[str, str] = {}
        while True:
            data = self.request(
                action="query",
                list="categorymembers",
                cmtitle=category,
                cmlimit="500",
                cmtype="page|subcat",
                **cont,
            )
            for item in data.get("query", {}).get("categorymembers", []):
                if item["ns"] == 0:
                    pages.append(item["title"])
                elif item["ns"] == 14 and not any(p in item["title"] for p in skip):
                    subcats.append(item["title"])
            if "continue" not in data:
                return pages, subcats
            cont = {"cmcontinue": data["continue"]["cmcontinue"]}

    def crawl(
        self,
        roots: Iterable[str],
        depth: int,
        max_titles: int,
        skip_patterns: Iterable[str] = (),
    ) -> dict[str, tuple[str, int]]:
        """Breadth-first category walk: title -> (category that first reached it, depth)."""

        skip = tuple(skip_patterns)
        found: dict[str, tuple[str, int]] = {}
        queue = deque((root, 0) for root in roots)
        seen = {root for root, _ in queue}
        while queue and len(found) < max_titles:
            category, level = queue.popleft()
            try:
                pages, subcats = self.category_members(category, skip)
            except ApiError as error:  # permanent (e.g. invalid title); transient ones propagate
                LOGGER.warning("Categoria ignorada %s: %s", category, error)
                self.skipped_categories.append(category)
                continue
            for title in pages:
                if title not in found and not title.startswith(("Lista de", "Listas de")):
                    found[title] = (category, level)
                    if len(found) >= max_titles:
                        break
            if level < depth:
                for sub in subcats:
                    if sub not in seen:
                        seen.add(sub)
                        queue.append((sub, level + 1))
        return found

    def _query_revisions(self, key: str, values: list[str]) -> dict[str, Any]:
        """One ``prop=revisions`` query, following continuation until every page has content.

        MediaWiki stops adding revisions once a response reaches its size limit and returns a
        ``continue`` marker; without following it, long articles would look missing.
        """

        params = {
            "action": "query",
            "prop": "revisions|pageprops",
            "rvprop": "ids|content",
            "rvslots": "main",
            "ppprop": "disambiguation",
            "redirects": "1",
            key: "|".join(values),
        }
        aliases: list[dict[str, str]] = []
        pages: dict[str, dict[str, Any]] = {}
        cont: dict[str, str] = {}
        while True:
            data = self.request(**params, **cont)
            query = data.get("query", {})
            if not cont:
                aliases = query.get("normalized", []) + query.get("redirects", [])
            for page in query.get("pages", []):
                merged = pages.setdefault(page["title"], dict(page))
                if page.get("revisions") and not merged.get("revisions"):
                    merged["revisions"] = page["revisions"]
                if page.get("pageprops"):
                    merged["pageprops"] = page["pageprops"]
            if "continue" not in data:
                break
            cont = {k: str(v) for k, v in data["continue"].items()}
        for title, page in pages.items():
            if not (page.get("missing") or page.get("invalid") or page.get("revisions")):
                LOGGER.warning("Página sem revisão após a continuação: %s", title)
        return {"aliases": aliases, "pages": list(pages.values())}

    @staticmethod
    def _page(page: dict[str, Any]) -> Page | None:
        revisions = page.get("revisions")
        if page.get("missing") or page.get("invalid") or not revisions:
            return None
        revision = revisions[0]
        return Page(
            title=page["title"],
            pageid=int(page["pageid"]),
            revid=int(revision["revid"]),
            wikitext=revision["slots"]["main"].get("content", ""),
            disambiguation="disambiguation" in (page.get("pageprops") or {}),
        )

    def fetch_pages(self, titles: list[str], batch: int = 50) -> dict[str, Page]:
        """Latest revisions by title, ``batch`` titles per request, keyed by requested title.

        Normalized or redirected titles map back to every title that was requested for the same
        page (a batch may hold both an article and a redirect to it).
        """

        out: dict[str, Page] = {}
        for start in range(0, len(titles), batch):
            chunk = titles[start : start + batch]
            result = self._query_revisions("titles", chunk)
            step = {item["from"]: item["to"] for item in result["aliases"]}
            requested_by: dict[str, list[str]] = {}
            for title in chunk:
                final, hops = title, 0
                while final in step and hops < 10:
                    final, hops = step[final], hops + 1
                requested_by.setdefault(final, []).append(title)
            for raw in result["pages"]:
                page = self._page(raw)
                if page is None:
                    continue
                for requested in requested_by.get(page.title, [page.title]):
                    out[requested] = page
        return out

    def fetch_revisions(self, revids: list[int], batch: int = 50) -> list[Page]:
        """Exact revisions by id, used to rebuild the corpus from ``manifest.csv``."""

        pages: list[Page] = []
        for start in range(0, len(revids), batch):
            chunk = [str(r) for r in revids[start : start + batch]]
            for raw in self._query_revisions("revids", chunk)["pages"]:
                page = self._page(raw)
                if page is not None:
                    pages.append(page)
        return pages


def _retry_after(headers: dict[str, str]) -> float:
    """Seconds requested by a numeric ``Retry-After`` header (0 when absent or a date)."""

    value = headers.get("Retry-After") or headers.get("retry-after") or ""
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0
