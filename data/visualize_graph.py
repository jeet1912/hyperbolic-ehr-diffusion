import argparse
import collections
import itertools
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from dataset import MimicCsvDataset

def group_split_indices(subject_ids: List[int], train_frac=0.8, val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(subject_ids)), dtype=np.int64)
    rng.shuffle(unique)

    n = len(unique)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_subj = set(unique[:n_train])
    val_subj = set(unique[n_train:n_train + n_val])
    test_subj = set(unique[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for i, s in enumerate(subject_ids):
        if s in train_subj:
            train_idx.append(i)
        elif s in val_subj:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


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
    parser = argparse.ArgumentParser(description="Visualize LLemr co-occurrence graph from task CSVs.")
    parser.add_argument("--task-csv", type=str, required=True,
                        help="LLemr task CSV (e.g., llemr_readmission_task.csv).")
    parser.add_argument("--cohort-csv", type=str, required=True,
                        help="LLemr cohort CSV for global subject splits.")
    parser.add_argument("--task-name", type=str, required=True,
                        choices=["mortality", "los", "readmission", "diagnosis"],
                        help="Task name for CSV loader.")
    parser.add_argument("--bin-hours", type=int, default=6,
                        help="Bin size in hours for CSV loader.")
    parser.add_argument("--drop-negative", action="store_true",
                        help="Drop events with negative timestamps.")
    parser.add_argument("--truncate", type=str, default="latest",
                        choices=["latest", "earliest"],
                        help="Truncate long sequences when loading CSVs.")
    parser.add_argument("--t-max", type=int, default=256,
                        help="Max visits for readmission/diagnosis when loading CSVs.")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "all"],
                        help="Which split to visualize (default: train).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subject-level split.")
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="Train split fraction.")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Validation split fraction.")
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

    dataset = MimicCsvDataset(
        task_csv=args.task_csv,
        cohort_csv=args.cohort_csv,
        task_name=args.task_name,
        bin_hours=args.bin_hours,
        drop_negative=args.drop_negative,
        truncate=args.truncate,
        t_max=args.t_max,
    )

    if args.split == "all":
        sequences = dataset.x
    else:
        train_idx, val_idx, test_idx = group_split_indices(
            dataset.subject_id,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        if args.split == "train":
            indices = train_idx
        elif args.split == "val":
            indices = val_idx
        else:
            indices = test_idx
        sequences = [dataset.x[i] for i in indices]

    inverted_code_map = {v: k for k, v in dataset.code_map.items()}

    graph = build_code_graph(sequences, top_n=args.top_n, min_edge_weight=args.min_edge_weight)
    visualize_graph(graph, inverted_code_map, args.output)
    print(f"Saved graph visualization to {args.output}")

if __name__ == "__main__":
    main()
