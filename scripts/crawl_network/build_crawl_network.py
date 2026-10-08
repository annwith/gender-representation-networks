"""Network of the category crawl of the ``corpus`` stage (listing phase only).

Replays, offline from the API cache (``data/raw/wiki``), the 24 breadth-first category walks of
``corpus.collect_candidates`` (8 theme roots and 16 extra categories) and records what they saw:

- category vertices: every category a walk opened (roots included);
- article vertices: every title a walk recorded;
- arcs ``category -> subcategory`` for every listing relation between opened categories, and
  ``category -> article`` for every title a walk went through while reading that category (in
  the category where a walk hit its title limit, only the titles before the cut).

``first`` marks the arcs the pipeline actually used: the parent that queued a subcategory and
the category that first recorded a title (``source_category`` in ``articles.jsonl``). Download
and cleaning are left out: a title is a vertex whether or not it was later fetched or kept.

The walk loop is a copy of ``WikiClient.crawl`` with bookkeeping; the script checks that it
records exactly the same titles as ``WikiClient.crawl`` and the same category, depth and themes
as ``data/corpus/articles.jsonl``.

Outputs (``data/crawl/``): ``crawl_network.graphml``, ``crawl_network.json`` (with a
precomputed layout: one radial tree per walk) and ``rede-de-coleta.html``, the interactive page
built from ``page_template.html`` with the JSON embedded. The page has a second tab, "Artigos",
with every candidate article of ``data/corpus/articles.jsonl`` (status, revision, paragraph and
word counts) and the clean text of the kept ones from ``paragraphs.jsonl``. The text goes in as
gzip-compressed, base64-encoded JSON (about 4.6 MB instead of 10 MB), which the browser inflates
on first use, so the page stays a single file that also works from ``file://``. A copy of the
page goes to ``docs/coleta/index.html`` for GitHub Pages.

A third tab, "Amostra", shows how the occurrence sample of the ``sample`` stage forms the
networks: counts by stratum and theme, the 80 core paragraphs token by token (vertex, cut by
``f_max`` or whitespace, from the Qwen3 tokenizer in the local cache), the quotas of the target
and control words and the 150 multitheme words, each with a concordance of its occurrences. It
reads ``outputs/experiment/<run>/sample/`` (``occurrences.csv`` and ``_manifest.json``) and
checks it against the corpus and the manifest before writing.

Usage::

    .venv/bin/python scripts/crawl_network/build_crawl_network.py
"""

from __future__ import annotations

import os

# the tokenizer comes from the local cache, like the API responses
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import base64
import csv
import gzip
import json
import math
import shutil
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import igraph as ig
import networkx as nx

from gender_networks.artifacts import RunPaths, read_jsonl
from gender_networks.plots import (
    CATEGORICAL,
    CATEGORY_LABELS,
    REASON_LABELS,
    REASON_ORDER,
    STRATUM_LABELS,
    STRATUM_ORDER,
    THEME_LABELS,
)
from gender_networks.sampling import load_tokenizer
from gender_networks.settings import CorpusSettings, load_settings
from gender_networks.tokens import CATEGORIES, tokenize_paragraph
from gender_networks.wiki import WikiClient

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "experiment.yaml"
DEFAULT_OUT = REPO / "data" / "crawl"
DEFAULT_DOCS = REPO / "docs" / "coleta"
TEMPLATE = Path(__file__).with_name("page_template.html")
DATA_MARK = "/*__DATA__*/"
ARTICLES_MARK = "/*__ARTICLES__*/"
TEXT_MARK = "/*__TEXT__*/"
SAMPLE_MARK = "/*__SAMPLE__*/"
KWIC_FIELDS = ["key", "stratum", "theme", "pageid", "left", "token", "right"]
KWIC_WINDOW = 70  # characters of context on each side of an occurrence
# state of a core token: whitespace (never a vertex), vertex, cut by f_max, vertex of a target
# or control word
WHITESPACE, VERTEX, CUT, STUDIED = 0, 1, 2, 3
ARTICLE_FIELDS = [
    "title", "theme", "origin", "source_category", "depth", "pageid", "revid", "reason",
    "themes_reached", "paragraphs", "words",
]  # fmt: skip


@dataclass(frozen=True)
class Walk:
    """One breadth-first walk, with the parameters of ``corpus.collect_candidates``."""

    walk_id: str  # "<theme>::root" or "<theme>::<extra category>", as the corpus groups
    theme: str
    origin: str
    roots: tuple[str, ...]
    depth: int
    max_titles: int


@dataclass
class WalkTrace:
    found: dict[str, tuple[str, int]] = field(default_factory=dict)
    opened: list[tuple[str, int]] = field(default_factory=list)  # in opening order
    tree_parent: dict[str, str] = field(default_factory=dict)  # category that queued it
    subcats: dict[str, list[str]] = field(default_factory=dict)
    read: dict[str, list[str]] = field(default_factory=dict)  # titles gone through
    n_pages: dict[str, int] = field(default_factory=dict)
    cut: str | None = None  # category where the title limit was reached
    unopened: int = 0  # categories still queued at the end


def walks(settings: CorpusSettings) -> list[Walk]:
    out = [
        Walk(f"{theme}::root", theme, "root", tuple(roots), settings.bfs_depth,
             settings.max_titles_per_theme)
        for theme, roots in settings.themes.items()
    ]  # fmt: skip
    for theme, categories in settings.extra_categories.items():
        for extra in categories:
            out.append(
                Walk(f"{theme}::{extra}", theme, extra, (extra,), min(1, settings.bfs_depth),
                     settings.articles_per_extra_category * 4)
            )  # fmt: skip
    return out


def trace_walk(client: WikiClient, walk: Walk, skip_patterns: Sequence[str]) -> WalkTrace:
    """``WikiClient.crawl`` with bookkeeping; the control flow is the same line by line."""

    skip = tuple(skip_patterns)
    trace = WalkTrace()
    found = trace.found
    queue = deque((root, 0) for root in walk.roots)
    seen = {root for root, _ in queue}
    while queue and len(found) < walk.max_titles:
        category, level = queue.popleft()
        pages, subcats = client.category_members(category, skip)
        trace.opened.append((category, level))
        trace.subcats[category] = subcats
        trace.n_pages[category] = len(pages)
        read: list[str] = []
        for title in pages:
            if title.startswith(("Lista de", "Listas de")):
                continue
            read.append(title)
            if title not in found:
                found[title] = (category, level)
                if len(found) >= walk.max_titles:
                    trace.cut = category
                    break
        trace.read[category] = read
        if level < walk.depth:
            for sub in subcats:
                if sub not in seen:
                    seen.add(sub)
                    queue.append((sub, level + 1))
                    trace.tree_parent[sub] = category
    trace.unopened = len(queue)
    return trace


def build_graph(walk_list: Sequence[Walk], traces: dict[str, WalkTrace]) -> nx.DiGraph:
    g = nx.DiGraph()
    theme_roots = {root for w in walk_list if w.origin == "root" for root in w.roots}
    extra_roots = {root for w in walk_list if w.origin != "root" for root in w.roots}

    # Categories: attributes from the first walk (in corpus order) that opened them.
    for walk in walk_list:
        trace = traces[walk.walk_id]
        for order, (category, level) in enumerate(trace.opened):
            if category not in g:
                kind = (
                    "theme_root" if category in theme_roots
                    else "extra_root" if category in extra_roots
                    else "category"
                )  # fmt: skip
                g.add_node(
                    category, label=category.removeprefix("Categoria:"), kind=kind,
                    theme=walk.theme, walk=walk.walk_id, depth=level, open_order=order,
                    n_pages=trace.n_pages[category], n_subcats=len(trace.subcats[category]),
                    walks=[], themes=[], cut=False, parent=trace.tree_parent.get(category),
                )  # fmt: skip
            node = g.nodes[category]
            node["walks"].append(walk.walk_id)
            if walk.theme not in node["themes"]:
                node["themes"].append(walk.theme)
        if trace.cut is not None:
            g.nodes[trace.cut]["cut"] = True

    # Articles: the corpus records the walk (group) that first reaches a title.
    reached: dict[str, set[str]] = {}
    for walk in walk_list:
        for order, (title, (category, level)) in enumerate(traces[walk.walk_id].found.items()):
            reached.setdefault(title, set()).add(walk.theme)
            if title not in g:
                g.add_node(
                    title, label=title, kind="article", theme=walk.theme, walk=walk.walk_id,
                    depth=level, first_category=category, walks=[], parent=category,
                    found_order=order,
                )  # fmt: skip
            g.nodes[title]["walks"].append(walk.walk_id)
    for title, themes in reached.items():
        g.nodes[title]["themes"] = sorted(themes)
        g.nodes[title]["multitheme"] = len(themes) > 1

    # Arcs, deduplicated over walks; ``first`` if any walk used that arc.
    for walk in walk_list:
        trace = traces[walk.walk_id]
        for category, _ in trace.opened:
            for sub in trace.subcats[category]:
                if sub in g:
                    first = trace.tree_parent.get(sub) == category
                    _add_arc(g, category, sub, "subcategory", first)
            for title in trace.read[category]:
                if title in g:
                    first = trace.found.get(title, (None,))[0] == category
                    _add_arc(g, category, title, "member", first)
    return g


def _add_arc(g: nx.DiGraph, u: str, v: str, kind: str, first: bool) -> None:
    if g.has_edge(u, v):
        g.edges[u, v]["first"] = g.edges[u, v]["first"] or first
    else:
        g.add_edge(u, v, kind=kind, first=first)


def check_against_pipeline(
    client: WikiClient,
    settings: CorpusSettings,
    walk_list: Sequence[Walk],
    traces: dict[str, WalkTrace],
    g: nx.DiGraph,
    articles_path: Path,
) -> int:
    for walk in walk_list:
        reference = client.crawl(
            walk.roots, walk.depth, walk.max_titles, settings.skip_category_patterns
        )
        if reference != traces[walk.walk_id].found:
            raise AssertionError(f"{walk.walk_id}: the traced walk differs from WikiClient.crawl")
    checked = 0
    with articles_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["origin"] is None:
                raise ValueError(
                    f"{articles_path} came from corpus-rebuild, which does not record the crawl "
                    "(origin, depth); rebuild the network from the articles.jsonl of `corpus`"
                )
            found = traces[f"{record['theme']}::{record['origin']}"].found
            expected = (record["source_category"], record["depth"])
            if found.get(record["title"]) != expected:
                got = found.get(record["title"])
                raise AssertionError(f"{record['title']}: {got} != {expected}")
            node = g.nodes[record["title"]]
            if node["themes"] != record["themes_reached"]:
                raise AssertionError(f"{record['title']}: themes {node['themes']}")
            if record["dropped_reason"] == "multi_theme" and not node["multitheme"]:
                raise AssertionError(f"{record['title']}: multi_theme drop but one theme")
            checked += 1
    return checked


def tree_root(g: nx.DiGraph, name: str) -> str:
    """Follow ``parent`` (the arc the pipeline used) up to a walk root."""

    while g.nodes[name]["parent"] is not None:
        name = g.nodes[name]["parent"]
    return name


def radial_tree(g: nx.DiGraph, members: list[str]) -> tuple[dict[str, tuple[float, float]], float]:
    """Reingold-Tilford circular layout of one tree: ring = depth, angle = discovery order.

    igraph places children in vertex-id order, so ``members`` (sorted by discovery) keeps the
    order in which the walk queued categories and recorded titles around the circle. The
    radius grows with the square root of the tree size, so every tree has a similar density.
    """

    index = {name: i for i, name in enumerate(members)}
    arcs = [(index[g.nodes[n]["parent"]], index[n]) for n in members if g.nodes[n]["parent"]]
    h = ig.Graph(n=len(members), edges=arcs, directed=True)
    coords = h.layout_reingold_tilford_circular(mode="out", root=[0]).coords
    radius = 12.0 * math.sqrt(len(members))
    reach = max((math.hypot(x, y) for x, y in coords), default=0.0) or 1.0
    scale = radius / reach
    return {n: (x * scale, y * scale) for n, (x, y) in zip(members, coords, strict=True)}, radius


def layout(g: nx.DiGraph, walk_list: Sequence[Walk]) -> dict[str, tuple[float, float]]:
    """One radial tree per walk; each theme tree with its extra-category trees below it.

    The 8 theme blocks sit on a 4 x 2 grid. Cross-tree arcs (a title listed by categories of
    several walks, multi-theme titles) are left to the drawing.
    """

    rank = {w.walk_id: i for i, w in enumerate(walk_list)}

    def discovery(name: str) -> tuple[int, int, int]:
        data = g.nodes[name]
        is_article = data["kind"] == "article"
        order = data["found_order"] if is_article else data["open_order"]
        return rank[data["walk"]], int(is_article), order

    trees: dict[str, list[str]] = {}
    for name in sorted(g.nodes, key=discovery):
        trees.setdefault(tree_root(g, name), []).append(name)

    gap = 40.0
    blocks = []
    for theme in dict.fromkeys(w.theme for w in walk_list):
        main = next(w.roots[0] for w in walk_list if w.theme == theme and w.origin == "root")
        extras = [w.roots[0] for w in walk_list if w.theme == theme and w.origin != "root"]
        pos, r_main = radial_tree(g, trees[main])
        laid = [radial_tree(g, trees[root]) for root in extras]
        if laid:
            r_max = max(r for _, r in laid)
            width = sum(2 * r for _, r in laid) + gap * (len(laid) - 1)
            x = -width / 2
            for extra_pos, r in laid:
                cx, cy = x + r, -(r_main + gap + r_max)
                pos.update({n: (px + cx, py + cy) for n, (px, py) in extra_pos.items()})
                x += 2 * r + gap
        blocks.append(pos)

    boxes = []
    for pos in blocks:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        boxes.append((min(xs), max(xs), min(ys), max(ys)))
    cell_w = max(b[1] - b[0] for b in boxes) + 2 * gap
    cell_h = max(b[3] - b[2] for b in boxes) + 2 * gap
    positions: dict[str, tuple[float, float]] = {}
    for i, (pos, (x0, x1, _, y1)) in enumerate(zip(blocks, boxes, strict=True)):
        col, row = i % 4, i // 4
        dx = col * cell_w + (cell_w - (x1 - x0)) / 2 - x0
        dy = -row * cell_h - gap - y1  # top of the block at the top of its cell
        positions.update({n: (x + dx, y + dy) for n, (x, y) in pos.items()})
    return positions


def to_json(
    g: nx.DiGraph,
    walk_list: Sequence[Walk],
    traces: dict[str, WalkTrace],
    positions: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    names = list(g.nodes)
    index = {name: i for i, name in enumerate(names)}
    nodes = []
    for name in names:
        data = g.nodes[name]
        x, y = positions[name]
        node = {
            "id": name,
            "label": data["label"],
            "kind": data["kind"],
            "theme": data["theme"],
            "themes": data["themes"],
            "walk": data["walk"],
            "walks": data["walks"],
            "depth": data["depth"],
            "parent": data["parent"],
            "x": round(x, 2),
            "y": round(y, 2),
        }
        if data["kind"] == "article":
            node["first_category"] = data["first_category"]
            node["multitheme"] = data["multitheme"]
        else:
            node.update(
                open_order=data["open_order"], n_pages=data["n_pages"],
                n_subcats=data["n_subcats"], cut=data["cut"],
            )  # fmt: skip
        nodes.append(node)
    edges = [
        [
            index[u], index[v], int(d["kind"] == "subcategory"), int(d["first"]),
            int(g.nodes[v]["parent"] == u),
        ]
        for u, v, d in g.edges(data=True)
    ]  # fmt: skip
    themes = list(dict.fromkeys(w.theme for w in walk_list))
    return {
        "themes": [
            {"id": t, "label": THEME_LABELS.get(t, t), "color": CATEGORICAL[i % len(CATEGORICAL)]}
            for i, t in enumerate(themes)
        ],
        "walks": [
            {
                "id": w.walk_id,
                "theme": w.theme,
                "origin": w.origin,
                "roots": list(w.roots),
                "depth": w.depth,
                "max_titles": w.max_titles,
                "titles": len(traces[w.walk_id].found),
                "opened": len(traces[w.walk_id].opened),
                "unopened": traces[w.walk_id].unopened,
                "cut": traces[w.walk_id].cut,
            }
            for w in walk_list
        ],  # fmt: skip
        "edge_fields": ["source", "target", "is_subcategory", "first", "tree"],
        "nodes": nodes,
        "edges": edges,
    }


def write_graphml(g: nx.DiGraph, positions: dict[str, tuple[float, float]], path: Path) -> None:
    """GraphML has no list type: lists become ``;``-joined strings; absent values become ""."""

    out = nx.DiGraph()
    keys = sorted({k for _, d in g.nodes(data=True) for k in d})
    for name, data in g.nodes(data=True):
        attrs: dict[str, Any] = {}
        for key in keys:
            value = data.get(key)
            if value is None:
                value = ""
            attrs[key] = ";".join(value) if isinstance(value, list) else value
        attrs["x"], attrs["y"] = positions[name]
        out.add_node(name, **attrs)
    out.add_edges_from(g.edges(data=True))
    nx.write_graphml(out, path, encoding="utf-8", prettyprint=True)


def articles_payload(articles_path: Path, paragraphs_path: Path) -> tuple[dict[str, Any], str]:
    """Rows of the "Artigos" tab and the compressed text of the kept articles.

    One row per record of ``articles.jsonl``, in corpus order. The text is a JSON list aligned
    with the rows (the kept paragraphs of each article, in order; empty for dropped ones),
    gzip-compressed with a fixed timestamp (so an unchanged corpus gives the same bytes) and
    base64-encoded.
    """

    texts: dict[int, list[str]] = {}
    for paragraph in read_jsonl(paragraphs_path):
        article = texts.setdefault(paragraph["pageid"], [])
        if paragraph["paragraph_idx"] != len(article):
            raise AssertionError(f"{paragraph['paragraph_id']}: paragraphs out of order")
        article.append(paragraph["text"])
    rows: list[list[Any]] = []
    text_rows: list[list[str]] = []
    for record in read_jsonl(articles_path):
        kept = record["dropped_reason"] is None
        paragraphs = texts.pop(record["pageid"], []) if kept else []
        if kept and not paragraphs:
            raise AssertionError(f"{record['title']}: kept without paragraphs")
        rows.append(
            [
                record["title"], record["theme"], record["origin"], record["source_category"],
                record["depth"], record["pageid"], record["revid"], record["dropped_reason"],
                record["themes_reached"], len(paragraphs),
                sum(len(text.split()) for text in paragraphs),
            ]
        )  # fmt: skip
        text_rows.append(paragraphs)
    if texts:
        raise AssertionError(f"paragraphs of {len(texts)} articles that are not kept")
    reasons = REASON_ORDER + sorted({r[7] for r in rows if r[7] and r[7] not in REASON_ORDER})
    labels = {**REASON_LABELS, "missing": "ausente"}
    table = {
        "fields": ARTICLE_FIELDS,
        "reasons": [{"id": r, "label": labels.get(r, r)} for r in reasons],
        "rows": rows,
    }
    return table, _pack(text_rows)


def _pack(data: Any) -> str:
    """Compact JSON, gzip-compressed with a fixed timestamp (same input, same bytes), base64."""

    packed = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(gzip.compress(packed, compresslevel=9, mtime=0)).decode("ascii")


def _context(text: str, start: int, end: int) -> tuple[str, str, str]:
    """Left context, occurrence and right context, cut at word boundaries."""

    left = text[max(0, start - KWIC_WINDOW) : start]
    right = text[end : end + KWIC_WINDOW]
    if start > KWIC_WINDOW and " " in left:
        left = "…" + left[left.index(" ") + 1 :]
    if end + KWIC_WINDOW < len(text) and " " in right:
        right = right[: right.rindex(" ")] + "…"
    return left, text[start:end], right


def sample_payload(
    sample_dir: Path, paragraphs_path: Path, tokenizer: Any, themes: Sequence[str]
) -> tuple[dict[str, Any], str]:
    """Data of the "Amostra" tab (summary counts and the gzip+base64 JSON the page inflates).

    Checks, before anything is written: every vertex is ``text[char_start:char_end]`` of its
    paragraph and starts a token of the same id; the core has the manifest's vertex and cut
    counts; the strata have the manifest's sizes; every target and control keeps its ``final``
    count in each theme; every multitheme type has ``before + drawn`` occurrences.
    """

    manifest = json.loads((sample_dir / "_manifest.json").read_text(encoding="utf-8"))
    with (sample_dir / "occurrences.csv").open(encoding="utf-8", newline="") as handle:
        occurrences = list(csv.DictReader(handle))
    needed = {row["paragraph_id"] for row in occurrences}
    paragraphs: dict[str, dict[str, Any]] = {}
    corpus_articles: set[int] = set()
    n_corpus_paragraphs = 0
    for p in read_jsonl(paragraphs_path):
        corpus_articles.add(p["pageid"])
        n_corpus_paragraphs += 1
        if p["paragraph_id"] in needed:
            paragraphs[p["paragraph_id"]] = p
    if n_corpus_paragraphs != manifest["corpus_paragraphs"]:
        raise AssertionError("the sample was drawn from another corpus")
    if len(paragraphs) != len(needed):
        raise AssertionError(f"{len(needed) - len(paragraphs)} sample paragraphs not in the corpus")
    for row in occurrences:
        text = paragraphs[row["paragraph_id"]]["text"]
        if text[int(row["char_start"]) : int(row["char_end"])] != row["token_text"]:
            raise AssertionError(f"occurrence {row['occurrence_id']}: offsets off the corpus text")

    # counts by stratum and theme: vertices, paragraphs, articles, types
    groups: dict[tuple[str, str], list[set[Any]]] = defaultdict(lambda: [set(), set(), set()])
    vertices: Counter[tuple[str, str]] = Counter()
    for row in occurrences:
        stratum, theme = row["stratum"], row["theme"]
        for key in ((stratum, theme), (stratum, ""), ("", theme), ("", "")):
            vertices[key] += 1
            paragraph, article, types = groups[key]
            paragraph.add(row["paragraph_id"])
            article.add(row["pageid"])
            types.add(row["token_id"])
    grid = {f"{s}|{t}": [vertices[(s, t)], *map(len, sets)] for (s, t), sets in groups.items()}
    by_stratum = {s: vertices[(s, "")] for s in STRATUM_ORDER if vertices[(s, "")]}
    if by_stratum != manifest["by_stratum"]:
        raise AssertionError(f"strata {by_stratum} != manifest {manifest['by_stratum']}")

    # core paragraphs token by token (position 0 of a sequence is the prefix token)
    core_rows: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in occurrences:
        if row["stratum"] == "core":
            core_rows[row["paragraph_id"]][int(row["pos_in_sequence"]) - 1] = row
    by_article: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for paragraph_id in core_rows:
        by_article[paragraphs[paragraph_id]["pageid"]].append(paragraphs[paragraph_id])
    core: list[dict[str, Any]] = []
    n_vertices = n_cut = 0
    for theme in themes:
        for pageid in manifest["core_articles"][theme]:
            items = []
            for paragraph in sorted(by_article.pop(pageid), key=lambda p: p["paragraph_idx"]):
                kept = core_rows[paragraph["paragraph_id"]]
                text = paragraph["text"]
                texts, states, cats, freqs = [], [], [], []
                shown = 0  # pieces of one character share its offsets; show it once
                for token in tokenize_paragraph(tokenizer, text):
                    row = kept.pop(token.index, None)
                    if row is not None:
                        if int(row["token_id"]) != token.token_id:
                            raise AssertionError(f"occurrence {row['occurrence_id']}: other token")
                        state = STUDIED if row["target_word"] else VERTEX
                        freq = int(row["f_sample"])
                        n_vertices += 1
                    elif token.category == "whitespace":
                        state, freq = WHITESPACE, 0
                    else:
                        state, freq = CUT, 0
                        n_cut += 1
                    texts.append(text[max(shown, token.char_start) : token.char_end])
                    shown = max(shown, token.char_end)
                    states.append(state)
                    cats.append(CATEGORIES.index(token.category))
                    freqs.append(freq)
                if kept or "".join(texts) != text:
                    raise AssertionError(f"{paragraph['paragraph_id']}: tokens off the corpus text")
                items.append(
                    {"idx": paragraph["paragraph_idx"],
                     "t": texts, "s": states, "c": cats, "f": freqs}
                )  # fmt: skip
            core.append({"pageid": pageid, "theme": theme, "paragraphs": items})
    if by_article:
        raise AssertionError(f"core paragraphs outside core_articles: {list(by_article)}")
    if (n_vertices, n_cut) != (manifest["core_vertices"], manifest["core_removed_by_cap"]):
        raise AssertionError(f"core: {n_vertices} vertices and {n_cut} cut, manifest differs")

    # target and control quotas, multitheme types, and the concordance of their occurrences
    final: Counter[tuple[str, str]] = Counter(
        (row["target_word"], row["sense_theme"]) for row in occurrences if row["target_word"]
    )
    words = []
    for kind, key in (("target", "targets"), ("control", "controls")):
        for word, info in manifest[key].items():
            quotas = []
            for theme, q in info["themes"].items():
                if final[(word, theme)] != q["final"]:
                    raise AssertionError(f"{word}/{theme}: {final[(word, theme)]} != {q['final']}")
                quotas.append([theme, q["requested"], q["core"], q["drawn"], q["final"]])
            # core occurrences outside the sense themes keep the word but fill no quota
            other = [
                [theme, final[(word, theme)]]
                for theme in themes
                if theme not in info["themes"] and final[(word, theme)]
            ]
            words.append(
                {"word": word, "kind": kind, "kept": info["kept"], "quotas": quotas, "other": other}
            )
    if sum(final.values()) != sum(q[4] for w in words for q in w["quotas"]) + sum(
        o[1] for w in words for o in w["other"]
    ):
        raise AssertionError("target or control occurrences outside the listed words")
    per_type = Counter(row["token_id"] for row in occurrences)
    multi = []
    for t in manifest["multitheme_types"]:
        if per_type[str(t["token_id"])] != t["before"] + t["drawn"]:
            raise AssertionError(f"multitheme {t['text']!r}: occurrences != before + drawn")
        multi.append(
            {"word": t["text"].strip(), "id": t["token_id"], "themes": t["themes"],
             "before": t["before"], "drawn": t["drawn"]}
        )  # fmt: skip
    multi_ids = {str(t["id"]) for t in multi}
    kwic = []
    for row in occurrences:
        if row["target_word"]:
            key = "w:" + row["target_word"]
        elif row["token_id"] in multi_ids:
            key = "m:" + row["token_id"]
        else:
            continue
        text = paragraphs[row["paragraph_id"]]["text"]
        left, token, right = _context(text, int(row["char_start"]), int(row["char_end"]))
        kwic.append([key, row["stratum"], row["theme"], int(row["pageid"]), left, token, right])

    sample = manifest["settings"]["sample"]
    data = {
        "f_max": sample["f_max"],
        "seed": sample["seed"],
        "core_design": sample["core"],
        "min_per_sense": sample["min_per_sense"],
        "multitheme_design": sample["multitheme"],
        "limits": sample["limits"],
        "totals": {
            "corpus_articles": len(corpus_articles),
            "corpus_paragraphs": n_corpus_paragraphs,
            "articles": len(groups[("", "")][1]),
            "paragraphs": len(groups[("", "")][0]),
            "vertices": manifest["n_vertices"],
            "types": manifest["n_types"],
            "tokens_to_process": manifest["tokens_to_process"],
            "core_paragraphs": manifest["core_paragraphs"],
            "core_before_cap": manifest["core_vertices_before_cap"],
            "core_cut": manifest["core_removed_by_cap"],
            "core_capped_types": manifest["core_capped_types"],
            "multitheme_eligible": manifest["multitheme_eligible"],
        },
        "strata": [{"id": s, "label": STRATUM_LABELS[s]} for s in STRATUM_ORDER],
        "categories": [CATEGORY_LABELS.get(c, "espaço") for c in CATEGORIES],
        "grid": grid,
        "core": core,
        "words": words,
        "dropped": manifest["dropped_targets"] + manifest["dropped_controls"],
        "multi": multi,
        "kwic_fields": KWIC_FIELDS,
        "kwic": kwic,
    }
    stats = {
        "vertices": manifest["n_vertices"], "core_paragraphs": len(core_rows),
        "core_vertices": n_vertices, "core_cut": n_cut, "words": len(words),
        "multitheme": len(multi), "concordance": len(kwic),
    }  # fmt: skip
    return stats, _pack(data)


def write_page(payloads: dict[str, str], path: Path) -> None:
    """Fill each mark of the page template (``</`` escaped so no tag closes early)."""

    page = TEMPLATE.read_text(encoding="utf-8")
    for mark, payload in payloads.items():
        if page.count(mark) != 1:
            raise ValueError(f"{TEMPLATE.name} must contain {mark} once")
        page = page.replace(mark, payload.replace("</", "<\\/"))
    path.write_text(page, encoding="utf-8")


def summary(g: nx.DiGraph, walk_list: Sequence[Walk], traces: dict[str, WalkTrace]) -> None:
    kinds = Counter(d["kind"] for _, d in g.nodes(data=True))
    arcs = Counter((d["kind"], d["first"]) for _, _, d in g.edges(data=True))
    multi = sum(1 for _, d in g.nodes(data=True) if d.get("multitheme"))
    several = sum(
        1 for n, d in g.nodes(data=True)
        if d["kind"] == "article"
        and sum(1 for u in g.predecessors(n)) > 1
    )  # fmt: skip
    print(f"vértices: {g.number_of_nodes()} {dict(kinds)}")
    print(f"arcos: {g.number_of_edges()} {dict(arcs)}")
    print(f"artigos multitema: {multi}; artigos em mais de uma categoria aberta: {several}")
    for walk in walk_list:
        t = traces[walk.walk_id]
        print(
            f"  {walk.walk_id:42s} títulos {len(t.found):4d}/{walk.max_titles:<4d} "
            f"abertas {len(t.opened):4d} na fila {t.unopened:4d} parou em {t.cut}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--docs", type=Path, default=DEFAULT_DOCS, help="Pasta da cópia para o GitHub Pages"
    )
    parser.add_argument(
        "--sample", type=Path, default=None,
        help="Pasta da etapa sample (padrão: a da execução do --config)",
    )  # fmt: skip
    args = parser.parse_args()

    settings = load_settings(args.config)
    sample_dir = args.sample or RunPaths.from_settings(settings, REPO).sample_dir
    if not (sample_dir / "occurrences.csv").exists():
        raise SystemExit(f"Falta a amostra em {sample_dir}: rode a etapa sample ou use --sample")
    corpus = settings.corpus
    client = WikiClient(
        REPO / settings.paths.raw_dir, corpus.user_agent, delay_s=0.0, offline=True,
        sleep=lambda _: None,
    )  # fmt: skip
    walk_list = walks(corpus)
    traces = {w.walk_id: trace_walk(client, w, corpus.skip_category_patterns) for w in walk_list}
    g = build_graph(walk_list, traces)
    checked = check_against_pipeline(
        client, corpus, walk_list, traces, g, REPO / settings.paths.corpus_dir / "articles.jsonl"
    )
    print(f"conferido: {len(walk_list)} buscas = WikiClient.crawl; {checked} artigos do corpus")
    summary(g, walk_list, traces)

    positions = layout(g, walk_list)
    args.out.mkdir(parents=True, exist_ok=True)
    graphml = args.out / "crawl_network.graphml"
    write_graphml(g, positions, graphml)
    back = nx.read_graphml(graphml)
    size = (g.number_of_nodes(), g.number_of_edges())
    if (back.number_of_nodes(), back.number_of_edges()) != size:
        raise AssertionError("GraphML read back with a different size")
    data = to_json(g, walk_list, traces, positions)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (args.out / "crawl_network.json").write_text(payload, encoding="utf-8")
    corpus_dir = REPO / settings.paths.corpus_dir
    table, text = articles_payload(corpus_dir / "articles.jsonl", corpus_dir / "paragraphs.jsonl")
    kept = sum(row[7] is None for row in table["rows"])
    print(f"artigos: {len(table['rows'])} candidatos, {kept} mantidos; texto {len(text):,} bytes")
    stats, sample = sample_payload(
        sample_dir, corpus_dir / "paragraphs.jsonl", load_tokenizer(settings.model),
        list(corpus.themes),
    )  # fmt: skip
    print(
        f"amostra: {stats['vertices']} vértices; núcleo {stats['core_paragraphs']} parágrafos, "
        f"{stats['core_vertices']} vértices, {stats['core_cut']} cortados por f_max; "
        f"{stats['words']} alvos e controles, {stats['multitheme']} multitema, "
        f"{stats['concordance']} linhas de concordância; {len(sample):,} bytes"
    )
    page = args.out / "rede-de-coleta.html"
    articles = json.dumps(table, ensure_ascii=False, separators=(",", ":"))
    write_page(
        {DATA_MARK: payload, ARTICLES_MARK: articles, TEXT_MARK: text, SAMPLE_MARK: sample}, page
    )
    args.docs.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(page, args.docs / "index.html")
    print(f"gravado: {graphml}, {args.out / 'crawl_network.json'}, {page} e {args.docs}/index.html")


if __name__ == "__main__":
    main()
