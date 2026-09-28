"""Polite, cached client for the Portuguese Wikipedia API.

Every response is cached on disk (gzip JSON keyed by the request parameters), so an interrupted
download resumes where it stopped and re-cleaning the corpus never touches the network again.
Network requests are spaced by ``delay_s`` and HTTP 429 responses honor ``Retry-After``; the
API rate-limited the feasibility study otherwise.
"""

from __future__ import annotations

import hashlib
import json
import logging
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
    """The API answered with an error (e.g. an invalid category); retrying will not help."""


@dataclass(frozen=True)
class Page:
    """Latest revision of an article."""

    title: str
    pageid: int
    revid: int
    wikitext: str


class WikiClient:
    """Minimal MediaWiki API client with on-disk cache, spacing and retries."""

    def __init__(
        self,
        cache_dir: Path,
        user_agent: str,
        delay_s: float = 2.5,
        api_url: str = API_URL,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 12,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.user_agent = user_agent
        self.delay_s = delay_s
        self.api_url = api_url
        self.transport = transport or urllib_transport
        self.sleep = sleep
        self.max_attempts = max_attempts
        self.network_requests = 0
        self.cache_hits = 0

    def _cache_path(self, params: dict[str, str]) -> Path:
        key = json.dumps(sorted(params.items()), ensure_ascii=False)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json.gz"

    def request(self, **params: str) -> dict[str, Any]:
        """GET the API with JSON output; cached responses never reach the network."""

        full = {"format": "json", "formatversion": "2", **{k: str(v) for k, v in params.items()}}
        path = self._cache_path(full)
        if path.exists():
            self.cache_hits += 1
            return read_json_gz(path)
        url = self.api_url + "?" + urllib.parse.urlencode({**full, "maxlag": "5"})
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "identity"}
        for attempt in range(1, self.max_attempts + 1):
            self.sleep(self.delay_s)
            self.network_requests += 1
            try:
                status, response_headers, body = self.transport(url, headers, 60.0)
            except OSError as error:
                wait = min(120.0, 5.0 * attempt)
                LOGGER.warning("Erro de rede (%s); nova tentativa em %.0f s", error, wait)
                self.sleep(wait)
                continue
            if status == 429 or status >= 500:
                retry_after = response_headers.get("Retry-After") or response_headers.get(
                    "retry-after"
                )
                backoff = min(600.0, 30.0 * 2 ** (attempt - 1))
                wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                LOGGER.warning("HTTP %s; aguardando %.0f s (tentativa %d)", status, wait, attempt)
                self.sleep(wait)
                continue
            if status != 200:
                raise RuntimeError(f"HTTP {status} for {url}")
            data = json.loads(body.decode("utf-8"))
            error = data.get("error", {})
            if error.get("code") == "maxlag":
                self.sleep(10.0)
                continue
            if error:
                raise ApiError(f"API error {error.get('code')}: {error.get('info')}")
            write_json_gz(path, data)
            return data
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
            except ApiError as error:
                LOGGER.warning("Categoria ignorada %s: %s", category, error)
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

    def _revisions(self, key: str, values: list[str]) -> list[Page]:
        pages: list[Page] = []
        data = self.request(
            action="query",
            prop="revisions",
            rvprop="ids|content",
            rvslots="main",
            redirects="1",
            **{key: "|".join(values)},
        )
        for page in data.get("query", {}).get("pages", []):
            revisions = page.get("revisions")
            if page.get("missing") or not revisions:
                continue
            revision = revisions[0]
            pages.append(
                Page(
                    title=page["title"],
                    pageid=int(page["pageid"]),
                    revid=int(revision["revid"]),
                    wikitext=revision["slots"]["main"].get("content", ""),
                )
            )
        return pages

    def fetch_pages(self, titles: list[str], batch: int = 50) -> dict[str, Page]:
        """Latest revisions by title, ``batch`` titles per request, keyed by requested title.

        Redirected or normalized titles map back to the title that was requested.
        """

        out: dict[str, Page] = {}
        for start in range(0, len(titles), batch):
            chunk = titles[start : start + batch]
            data = self.request(
                action="query",
                prop="revisions",
                rvprop="ids|content",
                rvslots="main",
                redirects="1",
                titles="|".join(chunk),
            )
            query = data.get("query", {})
            alias: dict[str, str] = {}
            for item in query.get("normalized", []) + query.get("redirects", []):
                alias[item["to"]] = alias.get(item["from"], item["from"])
            for page in query.get("pages", []):
                revisions = page.get("revisions")
                if page.get("missing") or not revisions:
                    continue
                revision = revisions[0]
                requested = alias.get(page["title"], page["title"])
                out[requested] = Page(
                    title=page["title"],
                    pageid=int(page["pageid"]),
                    revid=int(revision["revid"]),
                    wikitext=revision["slots"]["main"].get("content", ""),
                )
        return out

    def fetch_revisions(self, revids: list[int], batch: int = 50) -> list[Page]:
        """Exact revisions by id, used to rebuild the corpus from ``manifest.csv``."""

        pages: list[Page] = []
        for start in range(0, len(revids), batch):
            chunk = [str(r) for r in revids[start : start + batch]]
            pages.extend(self._revisions("revids", chunk))
        return pages
