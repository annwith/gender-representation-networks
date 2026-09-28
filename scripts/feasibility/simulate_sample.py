"""Simulate occurrence sampling from the thematic Wikipedia pool (feasibility study)."""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).parent
TARGETS = ["banco", "campo", "órgão", "estado", "carga", "rede", "nota", "capital",
           "família", "massa", "matriz", "planta", "corrente", "onda", "célula", "núcleo"]
CONTROLS = ["ano", "século", "grande", "primeiro", "importante", "água"]
STOP = set("""a à ao aos as às até com como da das de dela dele deles do dos e é em entre era
foi foram há isso isto já la lhe mais mas me mesmo muito na nas nem no nos o os ou para pela
pelas pelo pelos por qual quando que quem se sem ser seu seus sua suas também te tem têm ter
um uma umas uns são sobre após onde está estão sendo sido pode podem outro outra outros outras
seja ainda assim cada todo toda todos todas essa esse esses essas este esta estes estas aquele
aquela tal tais será seria desde durante bem apenas cerca vez parte""".split())
CUT = re.compile(r"^==+\s*(Referências|Ligações externas|Ver também|Bibliografia|Notas)", re.I | re.M)
BANDS = [(1, 2, "1–2"), (3, 9, "3–9"), (10, 49, "10–49"), (50, 10**9, "≥50")]


def paragraphs(text):
    match = CUT.search(text)
    text = text[: match.start()] if match else text
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("=") or len(line.split()) < 40:
            continue
        if not line.rstrip("\"»)”").endswith((".", "!", "?")):
            continue
        if sum(c.isdigit() for c in line) / len(line) > 0.08:
            continue
        yield line


def categorize(pieces):
    alnum = [any(ch.isalnum() for ch in piece) for piece in pieces]
    start = []
    for p, piece in enumerate(pieces):
        if not alnum[p]:
            start.append(None)
        else:
            start.append(p == 0 or piece[:1].isspace() or not pieces[p - 1][-1:].isalnum())
    labels = []
    for p in range(len(pieces)):
        if start[p] is None:
            labels.append("pontuação")
        elif start[p]:
            continued = p + 1 < len(pieces) and start[p + 1] is False
            labels.append("início de fragmentada" if continued else "palavra inteira")
        else:
            labels.append("continuação")
    return labels


def load_pool(path):
    rng = random.Random(438)
    by_theme, seen = defaultdict(list), set()
    for line in path.open(encoding="utf-8"):
        article = json.loads(line)
        for par in paragraphs(article["text"]):
            key = par[:80].lower()
            if key not in seen:
                seen.add(key)
                by_theme[article["theme"]].append(par)
    for theme in by_theme:
        rng.shuffle(by_theme[theme])
    pool = []
    for i in range(max(map(len, by_theme.values()))):  # round-robin across themes
        pool.extend((theme, by_theme[theme][i]) for theme in sorted(by_theme) if i < len(by_theme[theme]))
    return pool


class Tok:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        probe = self.tok.convert_ids_to_tokens(self.tok.encode(" casa", add_special_tokens=False))
        self.sentencepiece = probe[0].startswith("▁")

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def pieces(self, ids):
        if self.sentencepiece:
            return [p.replace("▁", " ") for p in self.tok.convert_ids_to_tokens(ids)]
        return [self.tok.decode([i]) for i in ids]


def sample(pool_tokens, order, f_max, goal, seed=7):
    """Add paragraphs in `order` until the capped vertex count reaches `goal`; then cap per type."""
    counts, vertices, chosen = Counter(), 0, []
    for index in order:
        chosen.append(index)
        for token_id in pool_tokens[index][0][1:]:  # first token of each sequence excluded
            vertices += counts[token_id] < f_max
            counts[token_id] += 1
        if vertices >= goal:
            break
    occ = [(tid, cat, pool_tokens[i][2]) for i in chosen
           for tid, cat in zip(pool_tokens[i][0][1:], pool_tokens[i][1][1:], strict=True)]
    by_type = defaultdict(list)
    for position, (tid, _, _) in enumerate(occ):
        by_type[tid].append(position)
    rng, keep = random.Random(seed), []
    for idx in by_type.values():
        keep.extend(idx if len(idx) <= f_max else rng.sample(idx, f_max))
    kept = [occ[i] for i in sorted(keep)]
    return len(chosen), len(occ), kept


def summarize(tok, kept, target_ids):
    ids = np.array([k[0] for k in kept]); cats = np.array([k[1] for k in kept]); themes = np.array([k[2] for k in kept])
    freq = Counter(ids.tolist())
    type_cat = {t: Counter(cats[ids == t].tolist()).most_common(1)[0][0] for t in freq}
    out = {"vértices": len(kept), "tipos": len(freq)}
    for lo, hi, name in BANDS:
        types = [t for t, c in freq.items() if lo <= c <= hi]
        out[f"tipos {name}"] = len(types)
        out[f"%vért {name}"] = round(100 * sum(freq[t] for t in types) / len(kept))
    out["%vért palavra inteira"] = round(100 * float(np.mean(cats == "palavra inteira")))
    out["tipos f≥3"] = sum(c >= 3 for c in freq.values())
    out["tipos f≥10"] = sum(c >= 10 for c in freq.values())
    out["tipos f≥10 palavra inteira"] = sum(c >= 10 and type_cat[t] == "palavra inteira" for t, c in freq.items())

    def is_content(t):
        s = tok.pieces([t])[0].strip()
        return type_cat[t] == "palavra inteira" and s.isalpha() and len(s) >= 3 and s.lower() not in STOP

    content = [t for t, c in freq.items() if c >= 10 and is_content(t)]
    spread = {t: Counter(themes[ids == t].tolist()) for t in content}
    multi = lambda n: [t for t in content if sum(v >= 3 for v in spread[t].values()) >= n]  # noqa: E731
    out["conteúdo f≥10"] = len(content)
    out["conteúdo f≥10, ≥3 oc. em ≥2 temas"] = len(multi(2))
    out["conteúdo f≥10, ≥3 oc. em ≥3 temas"] = len(multi(3))
    examples = ", ".join(tok.pieces([t])[0].strip() for t in sorted(multi(3), key=lambda t: -freq[t])[:25])
    targets = {}
    for word, tid in target_ids.items():
        per = Counter(themes[ids == tid].tolist())
        targets[word] = per
    return out, examples, targets


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
    tok = Tok(model)
    pool = load_pool(HERE / "wiki_articles.jsonl")
    theme_names = sorted({theme for theme, _ in pool})
    pool_tokens = []
    words = 0
    for theme, par in pool:
        ids = tok.encode(par)
        pool_tokens.append((ids, categorize(tok.pieces(ids)), theme))
        words += len(par.split())
    total = sum(len(p[0]) for p in pool_tokens)
    print(f"\n######## {model}  (sentencepiece={tok.sentencepiece})")
    print(f"pool: {len(pool)} parágrafos, {total} tokens, {total/words:.2f} tokens/palavra; "
          f"parágrafos por tema: {dict(Counter(t for t, _ in pool))}")

    target_ids = {}
    for word in TARGETS + CONTROLS:
        ids = tok.encode(" " + word)
        if len(ids) == 1:
            target_ids[word] = ids[0]
    print("palavras-alvo que NÃO são um token só:", [w for w in TARGETS + CONTROLS if w not in target_ids])

    natural = list(range(len(pool)))
    # Targeted: first, up to 12 paragraphs per (target word, theme) containing it; then natural order.
    rng = random.Random(11)
    contains = defaultdict(list)
    for index, (ids, _, theme) in enumerate(pool_tokens):
        for word, tid in target_ids.items():
            if word in TARGETS and tid in ids[1:]:
                contains[(word, theme)].append(index)
    first = []
    for key in sorted(contains):
        rng.shuffle(contains[key])
        first.extend(contains[key][:12])
    first = list(dict.fromkeys(first))
    first_set = set(first)
    targeted = first + [i for i in natural if i not in first_set]

    for f_max in (20, 50):
        for label, order in (("natural", natural), ("direcionada", targeted)):
            for goal in (10_000, 15_000):
                n_par, n_raw, kept = sample(pool_tokens, order, f_max, goal)
                if len(kept) < goal:
                    print(f"f_max={f_max} {label} meta {goal}: NÃO atingida (máx {len(kept)})")
                    continue
                if goal == 10_000:
                    print(f"f_max={f_max} {label}: 10k vértices ← {n_par} parágrafos / {n_raw} tokens")
                    continue
                out, examples, targets = summarize(tok, kept, target_ids)
                print(f"\n=== f_max={f_max}, amostragem {label}: 15k vértices ← {n_par} parágrafos / {n_raw} tokens")
                for key, value in out.items():
                    print(f"  {key}: {value}")
                print("  exemplos de conteúdo em ≥3 temas:", examples)
                ok = [w for w in TARGETS if w in targets and sum(v >= 10 for v in targets[w].values()) >= 2]
                print(f"  palavras-alvo com ≥10 ocorrências em ≥2 temas: {len(ok)} → {ok}")
                for word, per in targets.items():
                    print(f"    {word:11s} total={sum(per.values()):3d}  "
                          + " ".join(f"{t[:4]}={per.get(t, 0)}" for t in theme_names))

    print("\n=== pool inteiro (sem limite): ocorrências das palavras-alvo por tema")
    for word, tid in target_ids.items():
        per = Counter(theme for ids, _, theme in pool_tokens for x in ids[1:] if x == tid)
        print(f"  {word:11s} total={sum(per.values()):4d}  " + " ".join(f"{t[:4]}={per.get(t, 0)}" for t in theme_names))


if __name__ == "__main__":
    main()
