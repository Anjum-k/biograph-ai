"""Find biologically meaningful paths connecting a drug to a disease
(e.g. drug -> gene -> pathway -> disease) in a NetworkX view of the graph."""
import networkx as nx


def find_top_paths(graph_nx: nx.Graph, drug_node, disease_node, max_length=4, top_k=3):
    try:
        all_paths = list(
            nx.all_simple_paths(graph_nx, source=drug_node, target=disease_node, cutoff=max_length)
        )
    except nx.NetworkXNoPath:
        return []

    scored_paths = []
    for path in all_paths:
        weight = sum(
            graph_nx[path[i]][path[i + 1]].get("weight", 1.0) for i in range(len(path) - 1)
        )
        scored_paths.append((path, weight))

    scored_paths.sort(key=lambda x: x[1], reverse=True)
    return scored_paths[:top_k]
