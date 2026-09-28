import json
import urllib.parse
from pathlib import Path

from gender_networks.wiki import WikiClient


class FakeApi:
    """Transport answering from a function of the query parameters."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[dict[str, str]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float):
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        self.calls.append(params)
        status, body, response_headers = self.handler(params, len(self.calls))
        return status, response_headers, json.dumps(body).encode("utf-8")


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


def test_http_429_honors_retry_after(tmp_path: Path) -> None:
    def handler(params, n):
        if n == 1:
            return 429, {}, {"Retry-After": "7"}
        return 200, {"query": {}}, {}

    sleeps: list[float] = []
    wiki = client(tmp_path, FakeApi(handler), sleeps)

    assert wiki.request(action="query") == {"query": {}}
    assert sleeps == [0.5, 7.0, 0.5]


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
