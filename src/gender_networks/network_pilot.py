"""End-to-end token-occurrence network pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gender_networks.config import (
    ExperimentConfig,
    ExtractionConfig,
    ModelConfig,
    TokenizationConfig,
)
from gender_networks.modeling import extract_input_embeddings, load_model_and_tokenizer
from gender_networks.network_analysis import (
    GraphResult,
    build_union_knn_graph,
    detect_communities,
    graph_metrics,
    jaccard_per_occurrence,
    same_token_neighbor_fraction,
)
from gender_networks.tokenization import TokenizedPrompt, tokenize_prompt

DEFAULT_TEXT = (
    "O banco da praca abriu cedo. "
    "Mais tarde ela sentou no banco do jardim e leu sobre um banco de dados que "
    "guardava mapas. A rede do pescador secava ao sol; a rede de "
    "computadores da escola caiu quando a chuva chegou. Pedro comprou manga madura "
    "na feira, mas rasgou a manga da camisa durante o jogo. A capital recebeu "
    "visitantes, e a capital da empresa mudou de cidade. No fim, Ana voltou ao "
    "banco, ligou a rede e contou a historia para Pedro. Depois disso, descansou."
)


def _csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _decode(tokenizer: Any, ids: list[int]) -> str:
    try:
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(ids)


def make_occurrences(tokenizer: Any, tokenized: TokenizedPrompt) -> list[dict[str, Any]]:
    """Metadata preserves token occurrences, including tokenizer subtokens."""

    output: list[dict[str, Any]] = []
    for position, (token_id, token) in enumerate(zip(tokenized.input_ids, tokenized.tokens, strict=True)):
        start, end = max(0, position - 3), min(len(tokenized.input_ids), position + 4)
        output.append(
            {
                "position": position,
                "token_id": token_id,
                "token": token,
                "decoded_token": _decode(tokenizer, [token_id]),
                "context_start": start,
                "context_end_exclusive": end,
                "local_context": _decode(tokenizer, tokenized.input_ids[start:end]),
            }
        )
    return output


def select_layer_indices(hidden_state_count: int) -> dict[str, int]:
    """Select outputs of the first, middle, and last Transformer blocks."""

    blocks = hidden_state_count - 1
    if blocks < 3:
        raise ValueError(f"Pilot needs at least 3 blocks; model exposes only {blocks}.")
    return {"layer_initial": 1, "layer_middle": (blocks + 1) // 2, "layer_final": blocks}


def _forward_all(model: Any, tokenized: TokenizedPrompt) -> tuple[torch.Tensor, ...]:
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([tokenized.input_ids], dtype=torch.long, device=device)
    mask = torch.tensor([tokenized.attention_mask], dtype=torch.long, device=device)
    with torch.inference_mode():
        result = model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False, return_dict=True
        )
    if result.hidden_states is None:
        raise RuntimeError("Model returned no hidden states")
    return tuple(state[0].detach().to("cpu") for state in result.hidden_states)


def _plot_networks(
    graphs: dict[str, GraphResult], labels: dict[str, list[int]], rows: list[dict[str, Any]], output: Path
) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    layout = nx.spring_layout(graphs["lexical"].graph, seed=41)
    counts = Counter(row["token_id"] for row in rows)
    emphasized = [row["position"] for row in rows if counts[row["token_id"]] >= 3][:12]
    annotations = {
        position: f"{position}:{rows[position]['decoded_token'].strip() or rows[position]['token']}"
        for position in emphasized
    }
    colors = plt.get_cmap("tab20")
    for name, graph_result in graphs.items():
        figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
        nx.draw_networkx_edges(graph_result.graph, layout, edge_color="#94a3b8", alpha=0.30, ax=axis)
        nx.draw_networkx_nodes(
            graph_result.graph, layout, node_size=48,
            node_color=[colors(labels[name][node] % 20) for node in graph_result.graph.nodes],
            linewidths=0, ax=axis,
        )
        nx.draw_networkx_labels(graph_result.graph, layout, labels=annotations, font_size=6, ax=axis)
        axis.set_title(f"Rede {name.replace('_', ' ')} — layout lexical fixo")
        axis.set_axis_off()
        figure.savefig(directory / f"network_{name}.png", dpi=180)
        plt.close(figure)


def _plot_summaries(jaccard: dict[str, np.ndarray], dominance: dict[str, np.ndarray], output: Path) -> None:
    import matplotlib.pyplot as plt

    directory = output / "figures"
    names, scores = list(jaccard), list(jaccard.values())
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.boxplot(scores, tick_labels=[item.replace("_to_", " → ") for item in names], showmeans=True)
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Jaccard por ocorrência")
    axis.set_title("Reorganização das vizinhanças")
    axis.tick_params(axis="x", rotation=18)
    figure.savefig(directory / "jaccard_distribution.png", dpi=180)
    plt.close(figure)

    names, scores = list(dominance), list(dominance.values())
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(positions, [float(np.mean(item)) for item in scores], "o-", label="média")
    axis.plot(positions, [float(np.median(item)) for item in scores], "s-", label="mediana")
    axis.set_xticks(positions, [item.replace("layer_", "camada ") for item in names])
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("same_token_neighbor_fraction")
    axis.set_title("Dominância lexical")
    axis.legend()
    figure.savefig(directory / "lexical_dominance.png", dpi=180)
    plt.close(figure)


def _table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row.get(field, "")).replace("|", "\\|") for field in fields) + " |" for row in rows)
    return "\n".join(lines)


def _report(
    output: Path, model: str, rows: list[dict[str, Any]], layer_indices: dict[str, int],
    metrics: list[dict[str, Any]], jaccard: dict[str, np.ndarray], dominance: dict[str, np.ndarray],
    neighbors: list[dict[str, Any]],
) -> None:
    ids = [row["token_id"] for row in rows]
    repeated = {token_id for token_id, count in Counter(ids).items() if count > 1}
    selected = sorted(
        (position for position, token_id in enumerate(ids) if token_id in repeated),
        key=lambda position: (jaccard["lexical_to_layer_final"][position], position),
    )[:5]
    examples: list[dict[str, Any]] = []
    for position in selected:
        for network in ("lexical", "layer_initial", "layer_middle", "layer_final"):
            items = [row for row in neighbors if row["network"] == network and row["position"] == position]
            examples.append({
                "token": rows[position]["decoded_token"].strip() or rows[position]["token"],
                "posição": position, "camada": network,
                "5 vizinhos (posição: token)": "; ".join(
                    f"{item['neighbor_position']}:{item['neighbor_decoded_token'].strip() or item['neighbor_token']}" for item in items
                ),
            })
    metric_fields = [
        "network", "vertices", "edges", "average_degree", "density", "average_clustering",
        "global_clustering", "connected_components", "largest_component_size",
        "average_distance_largest_component", "diameter_largest_component", "community_count",
        "community_sizes", "modularity",
    ]
    jaccard_summary = [
        {"comparação": name.replace("_to_", " → "), "média": f"{np.mean(scores):.3f}", "mediana": f"{np.median(scores):.3f}", "mínimo": f"{np.min(scores):.3f}", "máximo": f"{np.max(scores):.3f}"}
        for name, scores in jaccard.items()
    ]
    dominance_summary = [
        {"rede": name, "média": f"{np.mean(scores):.3f}", "mediana": f"{np.median(scores):.3f}"}
        for name, scores in dominance.items()
    ]
    final_scores = jaccard["lexical_to_layer_final"]
    extremes = [
        {
            "grupo": "maior mudança (J baixo)",
            "posição": int(position),
            "token": rows[int(position)]["decoded_token"].strip() or rows[int(position)]["token"],
            "Jaccard lexical → final": f"{final_scores[int(position)]:.3f}",
            "contexto": rows[int(position)]["local_context"],
        }
        for position in np.argsort(final_scores)[:5]
    ]
    extremes.extend(
        {
            "grupo": "menor mudança (J alto)",
            "posição": int(position),
            "token": rows[int(position)]["decoded_token"].strip() or rows[int(position)]["token"],
            "Jaccard lexical → final": f"{final_scores[int(position)]:.3f}",
            "contexto": rows[int(position)]["local_context"],
        }
        for position in np.argsort(final_scores)[-5:][::-1]
    )
    report = f'''# Piloto: da rede lexical à rede contextual

## Resultado

Entraram **{len(rows)} ocorrências de tokens** na rede: **{len(set(ids))} token_ids distintos** e
{len(repeated)} tipos repetidos. O modelo foi `{model}`. Todos os grafos têm exatamente as mesmas
ocorrências como vértices; somente os vetores de representação mudam.

## Representações e camadas reais

`lexical` consulta diretamente `model.get_input_embeddings()` para cada `token_id`, de modo que
ocorrências com o mesmo id têm o mesmo vetor. Os estados contextuais vêm de uma única forward pass
com `output_hidden_states=True`. O índice 0 dessa tupla é a saída da pilha de embeddings e não é
usado como embedding lexical. Foram usadas as saídas dos blocos: inicial
`hidden_states[{layer_indices['layer_initial']}]`, intermediária
`hidden_states[{layer_indices['layer_middle']}]` e final
`hidden_states[{layer_indices['layer_final']}]`.

## Regra k-NN

Cada ocorrência escolhe 5 vizinhos por similaridade cosseno, sem self-loop. Empates são desfeitos
pela menor posição no texto; isso mantém explícitos os empates de cosseno 1 da rede lexical. O grafo
não direcionado usa a **união**: `i—j` existe quando `i` escolhe `j` ou `j` escolhe `i`. Os arquivos
de vizinhos preservam as escolhas direcionadas `N_i` de tamanho 5, usadas em Jaccard e dominância.

## Métricas globais

{_table(metrics, metric_fields)}

Distância média e diâmetro são calculados no maior componente. Comunidades usam modularidade gulosa
do NetworkX.

## Reorganização das vizinhanças

{_table(jaccard_summary, ['comparação', 'média', 'mediana', 'mínimo', 'máximo'])}

`tables/jaccard_per_occurrence.csv` contém a distribuição e permite ordenar as maiores/menores
mudanças; `figures/jaccard_distribution.png` mostra essa distribuição.

### Extremos lexical → final

{_table(extremes, ['grupo', 'posição', 'token', 'Jaccard lexical → final', 'contexto'])}

## Dominância lexical

{_table(dominance_summary, ['rede', 'média', 'mediana'])}

`same_token_neighbor_fraction` é a fração de `N_i` cujo `token_id` é igual ao da ocorrência central.

## Ocorrências repetidas com maior mudança lexical → final

Estes são tokens reais do tokenizer, inclusive subtokens. Veja `occurrences.csv` para o contexto.

{_table(examples, ['token', 'posição', 'camada', '5 vizinhos (posição: token)'])}

## Saídas

- `occurrences.csv`: posição, token/id, string decodificada e contexto local.
- `tables/neighbors.csv`, `network_metrics.csv`, `jaccard_per_occurrence.csv`,
  `lexical_dominance.csv` e `communities.csv`: resultados completos.
- `figures/network_*.png`: quatro redes no mesmo layout; cores indicam comunidades.
'''
    (output / "report.md").write_text(report, encoding="utf-8")


def run_pilot(model_name: str, output: Path, text: str, k: int = 5, device_map: str | None = "auto") -> Path:
    """Execute extraction, graph construction, metrics, graphics, and report."""

    config = ExperimentConfig(
        model=ModelConfig(name_or_path=model_name, device_map=device_map),
        tokenization=TokenizationConfig(add_special_tokens=False),
        extraction=ExtractionConfig(hidden_state_indices=(0,)),
    )
    bundle = load_model_and_tokenizer(config)
    tokenized = tokenize_prompt(bundle.tokenizer, text, config.tokenization)
    if not 100 <= len(tokenized.input_ids) <= 150:
        raise ValueError(f"Pilot text produced {len(tokenized.input_ids)} tokens; expected 100–150.")
    if not 1 <= k < len(tokenized.input_ids):
        raise ValueError("k must be between 1 and the number of occurrences minus one")
    output.mkdir(parents=True, exist_ok=True)
    rows = make_occurrences(bundle.tokenizer, tokenized)
    all_states = _forward_all(bundle.model, tokenized)
    layer_indices = select_layer_indices(len(all_states))
    representations = {"lexical": extract_input_embeddings(bundle.model, tokenized).numpy()}
    representations.update({name: all_states[index].numpy() for name, index in layer_indices.items()})
    graphs = {name: build_union_knn_graph(vectors, k) for name, vectors in representations.items()}
    token_ids = np.asarray(tokenized.input_ids)
    metrics: list[dict[str, Any]] = []
    labels: dict[str, list[int]] = {}
    community_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    dominance: dict[str, np.ndarray] = {}
    for name, result in graphs.items():
        label, summary = detect_communities(result.graph)
        labels[name] = label
        metrics.append({"network": name, **graph_metrics(result.graph), **summary})
        dominance[name] = same_token_neighbor_fraction(result.neighbors, token_ids)
        community_rows.extend({"network": name, "position": position, "community": value} for position, value in enumerate(label))
        for position, nearby in enumerate(result.neighbors):
            for rank, neighbor in enumerate(nearby, start=1):
                neighbor_row = rows[int(neighbor)]
                neighbor_rows.append({
                    "network": name, "position": position, "rank": rank, "neighbor_position": int(neighbor),
                    "similarity": f"{result.similarities[position, rank - 1]:.8f}",
                    "neighbor_token_id": neighbor_row["token_id"], "neighbor_token": neighbor_row["token"],
                    "neighbor_decoded_token": neighbor_row["decoded_token"], "neighbor_context": neighbor_row["local_context"],
                })
    comparisons = {
        "lexical_to_layer_initial": ("lexical", "layer_initial"),
        "lexical_to_layer_middle": ("lexical", "layer_middle"),
        "lexical_to_layer_final": ("lexical", "layer_final"),
        "layer_initial_to_layer_middle": ("layer_initial", "layer_middle"),
        "layer_middle_to_layer_final": ("layer_middle", "layer_final"),
    }
    jaccard = {name: jaccard_per_occurrence(graphs[a].neighbors, graphs[b].neighbors) for name, (a, b) in comparisons.items()}
    jaccard_rows = [
        {"comparison": name, "position": position, "jaccard": f"{score:.8f}", "token_id": rows[position]["token_id"], "token": rows[position]["token"], "local_context": rows[position]["local_context"]}
        for name, scores in jaccard.items() for position, score in enumerate(scores)
    ]
    dominance_rows = [
        {"network": name, "position": position, "same_token_neighbor_fraction": f"{score:.8f}", "token_id": rows[position]["token_id"], "token": rows[position]["token"], "local_context": rows[position]["local_context"]}
        for name, scores in dominance.items() for position, score in enumerate(scores)
    ]
    _csv(output / "occurrences.csv", rows, list(rows[0]))
    _csv(output / "tables/network_metrics.csv", metrics, list(metrics[0]))
    _csv(output / "tables/neighbors.csv", neighbor_rows, list(neighbor_rows[0]))
    _csv(output / "tables/jaccard_per_occurrence.csv", jaccard_rows, list(jaccard_rows[0]))
    _csv(output / "tables/lexical_dominance.csv", dominance_rows, list(dominance_rows[0]))
    _csv(output / "tables/communities.csv", community_rows, list(community_rows[0]))
    (output / "metadata.json").write_text(json.dumps({
        "model": model_name, "token_count": len(rows), "unique_token_ids": len(set(tokenized.input_ids)),
        "k": k, "hidden_state_count": len(all_states), "layer_indices": layer_indices,
        "graph_symmetrization": "undirected union of directed k-NN choices",
        "tie_breaking": "descending cosine, then ascending position",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_networks(graphs, labels, rows, output)
    _plot_summaries(jaccard, dominance, output)
    _report(output, model_name, rows, layer_indices, metrics, jaccard, dominance, neighbor_rows)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the complete token-occurrence network pilot.")
    parser.add_argument("--model", default="pierreguillou/gpt2-small-portuguese")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/network_pilot"))
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args(argv)
    run_pilot(args.model, args.output_dir, args.text, args.k, args.device_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
