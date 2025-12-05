import argparse
import collections
import itertools
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx


def load_mimic_sequences(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["x"], data["code_map"]


def build_code_graph(sequences, top_n: int = 100, min_edge_weight: int = 5):
    code_freq = collections.Counter()
    edge_weights = collections.Counter()

    for patient in sequences:
        for visit in patient:
            if not visit:
                continue
            visit_codes = sorted(set(visit))
            code_freq.update(visit_codes)
            for c1, c2 in itertools.combinations(visit_codes, 2):
                edge_weights[(c1, c2)] += 1

    most_common_codes = {code for code, _ in code_freq.most_common(top_n)}

    G = nx.Graph()
    for code in most_common_codes:
        G.add_node(code, frequency=code_freq[code])

    for (c1, c2), weight in edge_weights.items():
        if c1 in most_common_codes and c2 in most_common_codes and weight >= min_edge_weight:
            G.add_edge(c1, c2, weight=weight)

    return G


def visualize_graph(G, code_map, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(14, 10))
    degrees = dict(G.degree())
    nodes = list(G.nodes())
    node_sizes = [degrees[node] * 30 for node in nodes]
    node_colors = [degrees[node] for node in nodes]
    node_labels = {node: code_map.get(node, node) for node in G.nodes()}

    pos = nx.spring_layout(G, seed=42, k=0.3)
    scatter = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="viridis",
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    cbar = plt.colorbar(scatter, shrink=0.85)
    cbar.set_label("Node degree (co-occurring codes)", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize the MIMIC ICD9 graph from mimic_hf_cohort.pkl")
    parser.add_argument(
        "--pkl",
        type=str,
        default="data/mimic_hf_cohort.pkl",
        help="Path to mimic_hf_cohort pickle file",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=100,
        help="Number of most frequent codes to visualize",
    )
    parser.add_argument(
        "--min_edge_weight",
        type=int,
        default=5,
        help="Minimum co-occurrence count to draw an edge",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/plots/mimic_graph.png",
        help="Output path for the rendered graph image",
    )
    args = parser.parse_args()

    sequences, code_map = load_mimic_sequences(args.pkl)

    inverted_code_map = {v: k for k, v in code_map.items()}

    graph = build_code_graph(sequences, top_n=args.top_n, min_edge_weight=args.min_edge_weight)
    visualize_graph(graph, inverted_code_map, args.output)
    print(f"Saved graph visualization to {args.output}")

if __name__ == "__main__":
    main()
