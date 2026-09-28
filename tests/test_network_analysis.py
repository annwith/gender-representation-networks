import numpy as np

from gender_networks.network_analysis import (
    build_union_knn_graph,
    jaccard_per_occurrence,
    same_token_neighbor_fraction,
)


def test_knn_breaks_equal_lexical_embedding_ties_by_position() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    result = build_union_knn_graph(vectors, k=2)

    # Source 2 has three candidates tied at cosine 1 for positions 0 and 1;
    # the deterministic lower-position rule must choose 0 then 1.
    assert result.neighbors[2].tolist() == [0, 1]
    assert result.graph.has_edge(0, 2)


def test_neighbor_comparisons() -> None:
    left = np.array([[1, 2], [0, 2], [0, 1]])
    right = np.array([[1, 3], [0, 2], [1, 0]])

    assert jaccard_per_occurrence(left, right).tolist() == [1 / 3, 1.0, 1.0]
    fractions = same_token_neighbor_fraction(left, np.array([7, 7, 8]))
    assert fractions.tolist() == [0.5, 0.5, 0.0]
