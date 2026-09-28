"""Fetch a thematic sample of pt.wikipedia articles (feasibility study), politely rate-limited."""

import json
import re
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import mwparserfromhell

API = "https://pt.wikipedia.org/w/api.php"
UA = "gender-networks-research/0.1 (academic feasibility study)"
THEMES = {
    "fisica": "Categoria:Física",
    "biologia": "Categoria:Biologia",
    "economia": "Categoria:Economia",
    "politica": "Categoria:Política",
    "computacao": "Categoria:Ciência da computação",
    "musica": "Categoria:Música",
    "esporte": "Categoria:Futebol",
    "geografia": "Categoria:Geografia",
}
SKIP_CATS = ("Categoria:!", "Artigos", "Esboço", "Páginas", "Predefinições", "Wikipédia", "Usuário", "Listas")
OUT = Path(__file__).parent / "wiki_articles.jsonl"
PER_THEME = int(sys.argv[1]) if len(sys.argv) > 1 else 120
MAX_TITLES = 500
DELAY = 2.5
CUT = re.compile(r"^==+\s*(Referências|Ligações externas|Ver também|Bibliografia|Notas|Notas e referências|"
                 r"Leitura adicional|Leituras adicionais|Fontes)\s*==+", re.I | re.M)


def api(**params):
    params |= {"format": "json", "formatversion": "2", "maxlag": "5"}
    request = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    for attempt in range(6):
        time.sleep(DELAY)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
            if data.get("error", {}).get("code") == "maxlag":
                time.sleep(10)
                continue
            return data
        except urllib.error.HTTPError as error:
            wait = 60 if error.code == 429 else 5 * (attempt + 1)
            print(f"HTTP {error.code}; aguardando {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
        except Exception as error:  # noqa: BLE001
            print(f"erro {error}; nova tentativa", file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(params)


def members(category):
    pages, subcats, cont = [], [], {}
    while True:
        data = api(action="query", list="categorymembers", cmtitle=category, cmlimit="500",
                   cmtype="page|subcat", **cont)
        if "query" not in data:
            return pages, subcats
        for item in data["query"]["categorymembers"]:
            if item["ns"] == 0:
                pages.append(item["title"])
            elif item["ns"] == 14 and not any(bad in item["title"] for bad in SKIP_CATS):
                subcats.append(item["title"])
        if "continue" not in data:
            return pages, subcats
        cont = {"cmcontinue": data["continue"]["cmcontinue"]}


def collect(root, depth=2):
    seen, titles, frontier = {root}, [], [root]
    for _ in range(depth + 1):
        next_frontier = []
        for category in frontier:
            try:
                pages, subcats = members(category)
            except RuntimeError:
                continue
            titles.extend(pages)
            next_frontier.extend(c for c in subcats if c not in seen)
            seen.update(subcats)
            if len(titles) >= MAX_TITLES:
                return list(dict.fromkeys(titles))
        frontier = next_frontier
    return list(dict.fromkeys(titles))


def wikitexts(titles):
    """Up to 50 pages per request; returns {title: (pageid, revid, plaintext)}."""
    out = {}
    for i in range(0, len(titles), 50):
        data = api(action="query", prop="revisions", rvprop="ids|content", rvslots="main",
                   titles="|".join(titles[i:i + 50]), redirects="1")
        for page in data.get("query", {}).get("pages", []):
            revs = page.get("revisions")
            if not revs:
                continue
            raw = revs[0]["slots"]["main"]["content"]
            if "{{desambiguação" in raw.lower() or "{{desambig" in raw.lower():
                continue
            match = CUT.search(raw)
            raw = raw[: match.start()] if match else raw
            text = mwparserfromhell.parse(raw).strip_code(normalize=True, collapse=True)
            out[page["title"]] = (page["pageid"], revs[0]["revid"], raw, text)
    return out


def main():
    rng = random.Random(438)
    used, done = set(), set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            row = json.loads(line)
            used.add(row["title"]); done.add(row["theme"])
    with OUT.open("a", encoding="utf-8") as stream:
        for theme, root in THEMES.items():
            if theme in done:
                continue
            titles = [t for t in collect(root) if not t.startswith("Lista de") and t not in used]
            rng.shuffle(titles)
            fetched = wikitexts(titles[: int(PER_THEME * 1.3)])
            kept = 0
            for title, (pageid, revid, raw, text) in fetched.items():
                if kept >= PER_THEME or title in used:
                    continue
                used.add(title)
                stream.write(json.dumps({"theme": theme, "title": title, "pageid": pageid, "revid": revid,
                                         "text": text}, ensure_ascii=False) + "\n")
                kept += 1
            print(theme, "títulos:", len(titles), "mantidos:", kept, flush=True)


if __name__ == "__main__":
    main()
