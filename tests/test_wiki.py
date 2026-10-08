import gzip
import http.client
import json
import urllib.parse
from pathlib import Path

import pytest

from gender_networks.wiki import ApiError, CacheMiss, WikiClient


class FakeApi:
    """Transport answering from a function of the query parameters."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[dict[str, str]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float):
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        self.calls.append(params)
        status, body, response_headers = self.handler(params, len(self.calls))
        payload = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        return status, response_headers, payload


def client(tmp_path: Path, api: FakeApi, sleeps: list[float]) -> WikiClient:
    return WikiClient(tmp_path, "test-agent", delay_s=0.5, transport=api, sleep=sleeps.append)


def test_request_is_cached_on_disk(tmp_path: Path) -> None:
    api = FakeApi(lambda params, n: (200, {"query": {"ok": n}}, {}))
    sleeps: list[float] = []
    wiki = client(tmp_path, api, sleeps)

    first = wiki.request(action="query", titles="A")
    second = wiki.request(action="query", titles="A")

    assert first == second == {"query": {"ok": 1}}
    assert len(api.calls) == 1 and wiki.cache_hits == 1
    assert api.calls[0]["maxlag"] == "5" and api.calls[0]["formatversion"] == "2"
    assert sleeps == [0.5]  # polite spacing only before network requests


def test_http_429_waits_for_the_longer_of_retry_after_and_backoff(tmp_path: Path) -> None:
    def handler(params, n):
        if n == 1:
            return 429, {}, {"Retry-After": "90"}
        if n == 2:
            return 429, {}, {"Retry-After": "1"}  # a short value must not cut the backoff
        return 200, {"query": {}}, {}

    sleeps: list[float] = []
    wiki = client(tmp_path, FakeApi(handler), sleeps)

    assert wiki.request(action="query") == {"query": {}}
    assert sleeps == [0.5, 90.0, 0.5, 60.0, 0.5]


def test_repeated_429_without_retry_after_backs_off_exponentially(tmp_path: Path) -> None:
    def handler(params, n):
        return (429, {}, {}) if n <= 2 else (200, {"query": {}}, {})

    sleeps: list[float] = []
    wiki = client(tmp_path, FakeApi(handler), sleeps)

    wiki.request(action="query")

    assert sleeps == [0.5, 30.0, 0.5, 60.0, 0.5]


def test_category_members_follow_continuation_and_skip_maintenance(tmp_path: Path) -> None:
    def handler(params, n):
        if "cmcontinue" not in params:
            members = [
                {"ns": 0, "title": "Átomo"},
                {"ns": 14, "title": "Categoria:Física nuclear"},
                {"ns": 14, "title": "Categoria:!Artigos sobre física"},
                {"ns": 2, "title": "Usuário:Alguém/Rascunho"},
            ]
            return 200, {"query": {"categorymembers": members}, "continue": {"cmcontinue": "x"}}, {}
        return 200, {"query": {"categorymembers": [{"ns": 0, "title": "Elétron"}]}}, {}

    wiki = client(tmp_path, FakeApi(handler), [])

    pages, subcats = wiki.category_members("Categoria:Física", ["Categoria:!"])

    assert pages == ["Átomo", "Elétron"]
    assert subcats == ["Categoria:Física nuclear"]


def test_crawl_records_first_category_and_depth(tmp_path: Path) -> None:
    tree = {
        "Categoria:R": (["A", "Lista de coisas"], ["Categoria:S"]),
        "Categoria:S": (["A", "B"], ["Categoria:T"]),
        "Categoria:T": (["C"], []),
    }

    def handler(params, n):
        pages, subs = tree[params["cmtitle"]]
        members = [{"ns": 0, "title": t} for t in pages] + [{"ns": 14, "title": s} for s in subs]
        return 200, {"query": {"categorymembers": members}}, {}

    wiki = client(tmp_path, FakeApi(handler), [])

    found = wiki.crawl(["Categoria:R"], depth=1, max_titles=10)

    assert found == {"A": ("Categoria:R", 0), "B": ("Categoria:S", 1)}


def test_fetch_pages_maps_redirects_back_to_requested_titles(tmp_path: Path) -> None:
    def handler(params, n):
        body = {
            "query": {
                "redirects": [{"from": "Fisica", "to": "Física"}],
                "pages": [
                    {
                        "pageid": 1,
                        "title": "Física",
                        "revisions": [{"revid": 10, "slots": {"main": {"content": "texto"}}}],
                    },
                    {"title": "Inexistente", "missing": True},
                ],
            }
        }
        return 200, body, {}

    wiki = client(tmp_path, FakeApi(handler), [])

    pages = wiki.fetch_pages(["Fisica", "Inexistente"])

    assert list(pages) == ["Fisica"]
    assert (pages["Fisica"].pageid, pages["Fisica"].revid, pages["Fisica"].title) == (
        1,
        10,
        "Física",
    )


def test_5xx_and_maxlag_are_retried(tmp_path: Path) -> None:
    def handler(params, n):
        if n == 1:
            return 503, b"<html>down</html>", {}
        if n == 2:
            return 200, {"error": {"code": "maxlag", "info": "lag"}}, {}
        return 200, {"query": {"ok": True}}, {}

    sleeps: list[float] = []
    wiki = client(tmp_path, FakeApi(handler), sleeps)

    assert wiki.request(action="query") == {"query": {"ok": True}}
    assert sleeps == [0.5, 30.0, 0.5, 10.0, 0.5]


def test_incomplete_read_and_non_json_bodies_are_retried(tmp_path: Path) -> None:
    def handler(params, n):
        if n == 1:
            raise http.client.IncompleteRead(b"partial")
        if n == 2:
            return 200, b"<html>not json</html>", {}
        return 200, {"query": {}}, {}

    sleeps: list[float] = []
    wiki = client(tmp_path, FakeApi(handler), sleeps)

    assert wiki.request(action="query") == {"query": {}}
    assert sleeps == [0.5, 5.0, 0.5, 10.0, 0.5]


def test_transient_api_error_is_retried_not_raised(tmp_path: Path) -> None:
    def handler(params, n):
        if n == 1:
            return 200, {"error": {"code": "ratelimited", "info": "slow down"}}, {}
        return 200, {"query": {"ok": True}}, {}

    sleeps: list[float] = []
    wiki = client(tmp_path, FakeApi(handler), sleeps)

    assert wiki.request(action="query") == {"query": {"ok": True}}
    assert sleeps == [0.5, 30.0, 0.5]


def test_persistent_rate_limit_propagates_out_of_crawl(tmp_path: Path) -> None:
    api = FakeApi(lambda params, n: (429, {}, {}))
    wiki = WikiClient(
        tmp_path, "test-agent", delay_s=0.0, transport=api, sleep=lambda s: None, max_attempts=3
    )

    with pytest.raises(RuntimeError, match="Giving up"):
        wiki.crawl(["Categoria:R"], depth=1, max_titles=10)
    assert len(api.calls) == 3 and wiki.skipped_categories == []


def test_rate_limit_api_error_propagates_out_of_crawl(tmp_path: Path) -> None:
    api = FakeApi(lambda params, n: (200, {"error": {"code": "ratelimited"}}, {}))
    wiki = WikiClient(
        tmp_path, "test-agent", delay_s=0.0, transport=api, sleep=lambda s: None, max_attempts=2
    )

    with pytest.raises(RuntimeError, match="Giving up"):
        wiki.crawl(["Categoria:R"], depth=1, max_titles=10)


def test_permanent_api_error_skips_category_and_is_cached(tmp_path: Path) -> None:
    def handler(params, n):
        if params["cmtitle"] == "Categoria:Ruim":
            return 200, {"error": {"code": "invalidtitle", "info": "bad"}}, {}
        return 200, {"query": {"categorymembers": [{"ns": 0, "title": "A"}]}}, {}

    api = FakeApi(handler)
    wiki = client(tmp_path, api, [])

    assert wiki.crawl(["Categoria:Ruim", "Categoria:Boa"], 0, 10) == {"A": ("Categoria:Boa", 0)}
    assert wiki.skipped_categories == ["Categoria:Ruim"]

    offline = WikiClient(tmp_path, "test-agent", offline=True, transport=api)
    assert offline.crawl(["Categoria:Ruim", "Categoria:Boa"], 0, 10) == {"A": ("Categoria:Boa", 0)}
    assert len(api.calls) == 2  # the error response came from the cache
    with pytest.raises(ApiError):
        offline.request(
            action="query",
            list="categorymembers",
            cmtitle="Categoria:Ruim",
            cmlimit="500",
            cmtype="page|subcat",
        )


def test_offline_mode_raises_on_cache_miss(tmp_path: Path) -> None:
    api = FakeApi(lambda params, n: (200, {"query": {}}, {}))
    client(tmp_path, api, []).request(action="query", titles="A")
    offline = WikiClient(tmp_path, "test-agent", offline=True, transport=api)

    assert offline.request(action="query", titles="A") == {"query": {}}
    with pytest.raises(CacheMiss):
        offline.request(action="query", titles="B")
    assert len(api.calls) == 1


def test_truncated_cache_file_is_refetched_and_writes_are_atomic(tmp_path: Path) -> None:
    api = FakeApi(lambda params, n: (200, {"query": {"n": n}}, {}))
    wiki = client(tmp_path, api, [])
    wiki.request(action="query", titles="A")
    [path] = list(tmp_path.rglob("*.json.gz"))
    path.write_bytes(gzip.compress(b'{"query": {"n": 1}}')[:10])  # interrupted write

    assert wiki.request(action="query", titles="A") == {"query": {"n": 2}}
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == [path.name]


def page_json(pageid: int, title: str, **extra) -> dict:
    revision = {"revid": pageid * 10, "slots": {"main": {"content": f"texto {title}"}}}
    return {"pageid": pageid, "title": title, "revisions": [revision], **extra}


def test_fetch_pages_assigns_page_to_every_requested_alias(tmp_path: Path) -> None:
    def handler(params, n):
        body = {
            "query": {
                "normalized": [{"from": "física", "to": "Física"}],
                "redirects": [{"from": "Fisica", "to": "Física"}],
                "pages": [page_json(1, "Física")],
            }
        }
        return 200, body, {}

    wiki = client(tmp_path, FakeApi(handler), [])

    pages = wiki.fetch_pages(["Física", "Fisica", "física"])

    assert sorted(pages) == ["Fisica", "Física", "física"]
    assert {page.pageid for page in pages.values()} == {1}


def test_fetch_pages_follows_continuation_and_reads_disambiguation(tmp_path: Path) -> None:
    def handler(params, n):
        if "rvcontinue" not in params:
            pages = [page_json(1, "Longo"), {"pageid": 2, "title": "Cortado"}]
            return (
                200,
                {"query": {"pages": pages}, "continue": {"rvcontinue": "2|20", "continue": "||"}},
                {},
            )
        pages = [
            {"pageid": 1, "title": "Longo"},
            page_json(2, "Cortado", pageprops={"disambiguation": ""}),
        ]
        return 200, {"query": {"pages": pages}}, {}

    api = FakeApi(handler)
    wiki = client(tmp_path, api, [])

    pages = wiki.fetch_pages(["Longo", "Cortado"])

    assert sorted(pages) == ["Cortado", "Longo"]
    assert pages["Cortado"].disambiguation and not pages["Longo"].disambiguation
    assert pages["Longo"].wikitext == "texto Longo"
    assert api.calls[1]["rvcontinue"] == "2|20" and api.calls[1]["titles"] == "Longo|Cortado"


def test_fetch_revisions_batches_ids_and_skips_bad_revisions(tmp_path: Path) -> None:
    def handler(params, n):
        ids = [int(r) for r in params["revids"].split("|")]
        pages = [page_json(r // 10, f"Página {r}") for r in ids if r != 30]
        return 200, {"query": {"badrevids": {"30": {"revid": 30}}, "pages": pages}}, {}

    api = FakeApi(handler)
    wiki = client(tmp_path, api, [])

    pages = wiki.fetch_revisions([10, 20, 30], batch=2)

    assert [call["revids"] for call in api.calls] == ["10|20", "30"]
    assert [(page.pageid, page.revid) for page in pages] == [(1, 10), (2, 20)]
    assert pages[0].wikitext == "texto Página 10"
