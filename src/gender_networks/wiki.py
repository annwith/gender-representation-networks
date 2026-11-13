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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gender_networks.artifacts import read_json_gz, write_json_gz

LOGGER = logging.getLogger(__name__)
API_URL = "https://pt.wikipedia.org/w/api.php"

# transport(url, headers, timeout) -> (status, headers, body)
Transport = Callable[[str, dict[str, str], float], tuple[int, dict[str, str], bytes]]


def urllib_transport(url: str, headers: dict[str, str], timeout: float) -> tuple[int, dict, bytes]:
    """Default transport: one HTTP GET with ``urllib``.

    HTTP error statuses are returned, not raised, so :meth:`WikiClient.request` can decide
    whether to retry; connection failures still raise ``OSError``.

    Args:
        url: Full request URL, query string included.
        headers: Request headers.
        timeout: Socket timeout in seconds.

    Returns:
        ``(status, response headers, body bytes)``.
    """

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


@dataclass
class CrawlTrace:
    """What one :meth:`WikiClient.crawl` went through (filled when passed to it).

    Attributes:
        found: The crawl result: title to ``(category that first reached it, depth)``.
        opened: ``(category, depth)`` of every category whose members were listed, in order.
        tree_parent: Subcategory to the category that queued it.
        subcats: Category to its subcategories after the skip patterns.
        read: Category to the titles gone through, in order, up to the title limit.
        n_pages: Category to its number of article members (before the ``Lista de`` filter).
        cut: Category in which the title limit was reached; ``None`` when it never was.
        unopened: Categories still queued when the walk ended.
    """

    found: dict[str, tuple[str, int]] = field(default_factory=dict)  # what crawl returns
    opened: list[tuple[str, int]] = field(default_factory=list)  # categories, in opening order
    tree_parent: dict[str, str] = field(default_factory=dict)  # category that queued each one
    subcats: dict[str, list[str]] = field(default_factory=dict)
    read: dict[str, list[str]] = field(default_factory=dict)  # titles gone through, in order
    n_pages: dict[str, int] = field(default_factory=dict)
    cut: str | None = None  # category where the title limit was reached
    unopened: int = 0  # categories still queued at the end


@dataclass(frozen=True)
class Page:
    """Latest revision of an article.

    Also used for an exact revision fetched by id (:meth:`WikiClient.fetch_revisions`).

    Attributes:
        title: Page title as the API returns it (after normalization and redirects).
        pageid: Page id.
        revid: Id of the revision whose text is in ``wikitext``.
        wikitext: Content of the main slot; empty when the API returns none.
        disambiguation: True when the page has the ``disambiguation`` page property.
    """

    title: str
    pageid: int
    revid: int
    wikitext: str
    disambiguation: bool = False


class WikiClient:
    """Minimal MediaWiki API client with on-disk cache, spacing and retries.

    With ``offline=True`` a cache miss raises :class:`CacheMiss`, so a rebuild from the cache
    either reproduces the cached selection or fails loudly instead of fetching newer revisions.

    Attributes:
        cache_dir: Root of the response cache.
        user_agent: ``User-Agent`` header of every request.
        delay_s: Seconds slept before every network request.
        api_url: API endpoint.
        transport: Function that performs the HTTP request (replaced in the tests).
        sleep: Function used to wait (replaced in the tests).
        max_attempts: Network attempts per request before giving up.
        offline: Forbid network requests.
        network_requests: Number of network attempts made so far.
        cache_hits: Number of requests answered from the cache so far.
        skipped_categories: Categories :meth:`crawl` skipped after a permanent API error.
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
        """Create a client; nothing is read or requested yet.

        Args:
            cache_dir: Root of the response cache; created on the first write.
            user_agent: ``User-Agent`` header (Wikimedia asks for a contact in it).
            delay_s: Seconds slept before every network request. Defaults to 2.5.
            api_url: API endpoint. Defaults to the Portuguese Wikipedia.
            transport: ``transport(url, headers, timeout) -> (status, headers, body)``; ``None``
                uses :func:`urllib_transport`.
            sleep: Function called to wait, with the seconds as argument.
            max_attempts: Network attempts per request. Defaults to 12.
            offline: Raise :class:`CacheMiss` instead of requesting. Defaults to False.
        """

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
        """Cache file of a request.

        Args:
            params: Complete request parameters (``format`` and ``formatversion`` included,
                ``maxlag`` not).

        Returns:
            ``cache_dir/<first two hex digits>/<sha256>.json.gz``, where the hash is taken over
            the parameters sorted by name, so the key does not depend on argument order.
        """

        key = json.dumps(sorted(params.items()), ensure_ascii=False)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json.gz"

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        """Cached response, or None (a corrupt file from an interrupted write is deleted).

        Args:
            path: Cache file from :meth:`_cache_path`.

        Returns:
            The decoded response, or ``None`` when the file is absent or unreadable.

        Side effects:
            Deletes an unreadable cache file and logs a warning.
        """

        if not path.exists():
            return None
        try:
            return read_json_gz(path)
        except (EOFError, OSError, ValueError) as error:
            LOGGER.warning("Cache corrompido %s (%s); será baixado de novo", path, error)
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, path: Path, data: dict[str, Any]) -> None:
        """Atomic write: a killed process never leaves a truncated cache file behind.

        Args:
            path: Final cache file.
            data: Decoded API response to store.

        Side effects:
            Writes a temporary file named after the process id next to ``path`` and renames it
            over ``path``.
        """

        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        write_json_gz(temporary, data)
        os.replace(temporary, path)

    @staticmethod
    def _checked(data: dict[str, Any]) -> dict[str, Any]:
        """Return a response, or raise when it carries an API error.

        Args:
            data: Decoded API response.

        Returns:
            ``data`` unchanged.

        Raises:
            ApiError: If the response has an ``error`` object. Only permanent errors reach this
                point, because transient ones are retried and never cached.
        """

        error = data.get("error")
        if error:
            raise ApiError(f"API error {error.get('code')}: {error.get('info')}")
        return data

    def request(self, **params: str) -> dict[str, Any]:
        """GET the API with JSON output; cached responses never reach the network.

        Permanent API errors are cached too, so a rerun walks the category tree identically.

        On a cache miss the request is attempted up to ``max_attempts`` times. Every attempt
        first sleeps ``delay_s``. Network and decoding errors wait ``min(120, 5 * attempt)``
        seconds; HTTP 429 and 5xx wait the larger of ``Retry-After`` and an exponential backoff
        (30 s doubling up to 600 s); a ``maxlag`` error waits at least 10 s; any other API error
        code outside :data:`PERMANENT_ERRORS` waits the backoff.

        Args:
            **params: API parameters; values are converted with ``str``. ``format=json`` and
                ``formatversion=2`` are added, and ``maxlag=5`` is sent but is not part of the
                cache key.

        Returns:
            The decoded response.

        Raises:
            CacheMiss: In offline mode, when the response is not cached.
            ApiError: If the API answers with a permanent error (also when read from the cache).
            RuntimeError: On another HTTP status (403, 404...), or after ``max_attempts``
                attempts.

        Side effects:
            Sleeps, performs network requests, writes the cache and updates ``cache_hits`` and
            ``network_requests``.
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
        """Articles (namespace 0) and content subcategories (namespace 14) of a category.

        Args:
            category: Category title with its prefix (``Categoria:Física``).
            skip_patterns: Substrings; a subcategory whose title contains one is left out. The
                filter is applied after the request, so it does not change the cache key.

        Returns:
            ``(article titles, subcategory titles)`` in API order (the category sort key, in
            general alphabetical), following continuation until the list is complete.

        Raises:
            ApiError: If the category title is invalid.
        """

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
        max_titles: int | None,
        skip_patterns: Iterable[str] = (),
        trace: CrawlTrace | None = None,
    ) -> dict[str, tuple[str, int]]:
        """Breadth-first category walk: title -> (category that first reached it, depth).

        The API lists members alphabetically, so stopping at ``max_titles`` keeps the first
        subcategories and titles of the alphabet; ``None`` walks the whole tree to ``depth``.
        ``trace``, when given, records the walk (for the crawl page of ``scripts/``).

        Args:
            roots: Categories the walk starts from, all at depth 0.
            depth: Deepest category level opened; members of a category at this level are read,
                its subcategories are not queued.
            max_titles: Stop as soon as this many distinct titles were found; ``None`` for no
                limit.
            skip_patterns: Substrings of subcategory titles that are not queued.
            trace: Object filled with the details of the walk.

        Returns:
            Title to ``(category, depth of that category)``, in discovery order. Titles starting
            with ``Lista de`` or ``Listas de`` are left out.

        Side effects:
            A category that answers with a permanent API error is logged, appended to
            ``skipped_categories`` and skipped. Progress is logged every 200 categories.
        """

        skip = tuple(skip_patterns)
        roots = tuple(roots)
        found: dict[str, tuple[str, int]] = {}
        queue = deque((root, 0) for root in roots)
        seen = set(roots)
        opened = 0
        while queue and (max_titles is None or len(found) < max_titles):
            category, level = queue.popleft()
            opened += 1
            if opened % 200 == 0:
                LOGGER.info(
                    "Busca a partir de %s: %d categorias abertas, %d na fila, %d títulos",
                    ", ".join(roots), opened, len(queue), len(found),
                )
            try:
                pages, subcats = self.category_members(category, skip)
            except ApiError as error:  # permanent (e.g. invalid title); transient ones propagate
                LOGGER.warning("Categoria ignorada %s: %s", category, error)
                self.skipped_categories.append(category)
                continue
            read: list[str] = []
            for title in pages:
                if title.startswith(("Lista de", "Listas de")):
                    continue
                read.append(title)
                if title not in found:
                    found[title] = (category, level)
                    if max_titles is not None and len(found) >= max_titles:
                        if trace is not None:
                            trace.cut = category
                        break
            if trace is not None:
                trace.opened.append((category, level))
                trace.subcats[category] = subcats
                trace.n_pages[category] = len(pages)
                trace.read[category] = read
            if level < depth:
                for sub in subcats:
                    if sub not in seen:
                        seen.add(sub)
                        queue.append((sub, level + 1))
                        if trace is not None:
                            trace.tree_parent[sub] = category
        if trace is not None:
            trace.found = found
            trace.unopened = len(queue)
        return found

    def _query_revisions(self, key: str, values: list[str]) -> dict[str, Any]:
        """One ``prop=revisions`` query, following continuation until every page has content.

        MediaWiki stops adding revisions once a response reaches its size limit and returns a
        ``continue`` marker; without following it, long articles would look missing.

        Args:
            key: ``titles`` or ``revids``.
            values: Titles or revision ids (as strings) of one batch, joined with ``|``.

        Returns:
            ``{"aliases": [...], "pages": [...]}``: the ``normalized`` and ``redirects`` entries
            of the first response (``{"from", "to"}`` pairs), and one merged page object per
            title, keeping the first revision list seen for it.

        Side effects:
            Logs a warning for a page that is neither missing nor invalid and still has no
            revision after the continuation.
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
        """:class:`Page` of an API page object, or ``None`` when it has no usable revision.

        Args:
            page: Page object of a ``prop=revisions`` response.

        Returns:
            The page built from the first revision in the object; ``None`` for missing or
            invalid pages and for pages without revisions.
        """

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

        Args:
            titles: Titles to fetch. The batches are consecutive slices of this list, and each
                batch is one cache key, so the same list in the same order hits the cache.
            batch: Titles per request. Defaults to 50, the API maximum.

        Returns:
            Requested title to its :class:`Page`; titles that are missing or invalid are absent.
            Redirect chains are followed for at most 10 hops.
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
        """Exact revisions by id, used to rebuild the corpus from ``manifest.csv``.

        Args:
            revids: Revision ids to fetch.
            batch: Ids per request. Defaults to 50.

        Returns:
            One :class:`Page` per revision the API still serves, in response order (not in the
            order of ``revids``); callers index the result by ``revid``.
        """

        pages: list[Page] = []
        for start in range(0, len(revids), batch):
            chunk = [str(r) for r in revids[start : start + batch]]
            for raw in self._query_revisions("revids", chunk)["pages"]:
                page = self._page(raw)
                if page is not None:
                    pages.append(page)
        return pages


def _retry_after(headers: dict[str, str]) -> float:
    """Seconds requested by a numeric ``Retry-After`` header (0 when absent or a date).

    Args:
        headers: Response headers; the name is looked up as ``Retry-After`` and
            ``retry-after`` only.

    Returns:
        Non-negative number of seconds.
    """

    value = headers.get("Retry-After") or headers.get("retry-after") or ""
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0
