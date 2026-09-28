"""Hybrid design: dense core of whole paragraphs + sparse strata of selected occurrences."""

import random
import sys
from collections import Counter, defaultdict

import numpy as np

from simulate_sample import CONTROLS, HERE, STOP, TARGETS, Tok, categorize, load_pool

F_MAX = 50
GOAL = 15_000
CONTENT_TYPES = 150
CONTENT_QUOTA = 20


def main():
    model = sys.argv[1]
    tok = Tok(model)
    pool = load_pool(HERE / "wiki_articles.jsonl")
    themes = sorted({t for t, _ in pool})
    paras = []
    for theme, par in pool:
        ids = tok.encode(par)
        paras.append((ids, categorize(tok.pieces(ids)), theme))

    # Index every occurrence (excluding sequence position 0) by type.
    where = defaultdict(lambda: defaultdict(list))  # type -> theme -> [(par, pos)]
    for pi, (ids, cats, theme) in enumerate(paras):
        for pos in range(1, len(ids)):
            where[ids[pos]][theme].append((pi, pos))
    cat_of = {}
    for pi, (ids, cats, _) in enumerate(paras):
        for pos in range(1, len(ids)):
            cat_of.setdefault(ids[pos], Counter())[cats[pos]] += 1
    main_cat = {t: c.most_common(1)[0][0] for t, c in cat_of.items()}

    def text(t):
        return tok.pieces([t])[0].strip()

    def is_content(t):
        s = text(t)
        return main_cat[t] == "palavra inteira" and s.isalpha() and len(s) >= 3 and s.lower() not in STOP

    single = {w: tok.encode(" " + w)[0] for w in TARGETS + CONTROLS if len(tok.encode(" " + w)) == 1}
    rng = random.Random(438)
    chosen = {}  # (par, pos) -> stratum
    count = Counter()

    def take_balanced(t, quota, stratum):
        buckets = {th: rng.sample(v, len(v)) for th, v in where[t].items()}
        while count[t] < quota and any(buckets.values()):
            for th in themes:
                if count[t] >= quota:
                    break
                while buckets.get(th):
                    occ = buckets[th].pop()
                    if occ not in chosen:
                        chosen[occ] = stratum
                        count[t] += 1
                        break

    for word, t in single.items():
        take_balanced(t, F_MAX, "alvo" if word in TARGETS else "controle")
    eligible = [t for t in where if t not in single.values() and is_content(t)
                and sum(len(v) >= 5 for v in where[t].values()) >= 3]
    content = rng.sample(eligible, min(CONTENT_TYPES, len(eligible)))
    for t in content:
        take_balanced(t, CONTENT_QUOTA, "conteúdo multitema")
    sparse = len(chosen)

    # Dense core: whole paragraphs (round-robin across themes), capped per type, until GOAL.
    core_pars = 0
    for pi, (ids, _, _) in enumerate(paras):
        if len(chosen) >= GOAL:
            break
        core_pars += 1
        for pos in range(1, len(ids)):
            if (pi, pos) not in chosen and count[ids[pos]] < F_MAX:
                chosen[(pi, pos)] = "núcleo"
                count[ids[pos]] += 1

    occ = list(chosen)
    ids = np.array([paras[p][0][q] for p, q in occ]); th = np.array([paras[p][2] for p, _ in occ])
    stratum = Counter(chosen.values())
    touched = {p for p, _ in occ}
    last = defaultdict(int)
    for p, q in occ:
        last[p] = max(last[p], q)
    to_run = sum(v + 1 for v in last.values())
    freq = Counter(ids.tolist())
    print(f"\n######## {model}: desenho híbrido (f_max={F_MAX})")
    print(f"vértices: {len(occ)}  por estrato: {dict(stratum)}")
    print(f"núcleo denso: {core_pars} parágrafos inteiros; parágrafos tocados no total: {len(touched)}; "
          f"tokens a processar no modelo (até o último vértice de cada parágrafo): {to_run}")
    print(f"tipos: {len(freq)}; tipos com f 1–2: {sum(c <= 2 for c in freq.values())} "
          f"({100*sum(c for c in freq.values() if c <= 2)/len(occ):.0f}% dos vértices); "
          f"f≥3: {sum(c >= 3 for c in freq.values())}; f≥10: {sum(c >= 10 for c in freq.values())}")
    print(f"palavras de conteúdo elegíveis para o estrato multitema no pool: {len(eligible)}")
    cont10 = [t for t, c in freq.items() if c >= 10 and is_content(t)]
    spread = {t: Counter(th[ids == t].tolist()) for t in cont10}
    for n in (2, 3):
        print(f"conteúdo f≥10 com ≥3 ocorrências em ≥{n} temas: "
              f"{sum(sum(v >= 3 for v in spread[t].values()) >= n for t in cont10)}")
    for label, words in (("alvo", TARGETS), ("controle", CONTROLS)):
        ok2 = [w for w in words if w in single and sum(v >= 10 for v in Counter(th[ids == single[w]].tolist()).values()) >= 2]
        ok3 = [w for w in words if w in single and sum(v >= 5 for v in Counter(th[ids == single[w]].tolist()).values()) >= 3]
        print(f"{label}: ≥10 oc. em ≥2 temas: {len(ok2)}/{len([w for w in words if w in single])} {ok2}")
        print(f"{label}: ≥5 oc. em ≥3 temas:  {len(ok3)} {ok3}")
    core_idx = [i for i, o in enumerate(occ) if chosen[o] == "núcleo"]
    per_par = Counter(occ[i][0] for i in core_idx)
    print(f"núcleo: {len(core_idx)} vértices, média {np.mean(list(per_par.values())):.0f} vértices por parágrafo")


if __name__ == "__main__":
    main()
