import networkx as nx
import random
import os
import matplotlib.pyplot as plt

class ToyICDHierarchy:
    """
    Build a toy ICD-like tree:
    - 5 root chapters
    - each with 5 children
    - each child with 5 leaf codes
    => 5 + 25 + 125 = 155 codes (base)
    """
    def __init__(self, seed: int = 42, extra_depth: int = 0):
        random.seed(seed)
        self.G = nx.DiGraph()
        self.codes = []
        self.extra_depth = extra_depth
        # roots
        for i in range(5):
            root = f"C{i}"
            self.G.add_node(root, depth=0)
            self.codes.append(root)
            # 1st level
            for j in range(5):
                mid = f"C{i}{j}"
                self.G.add_node(mid, depth=1)
                self.G.add_edge(root, mid)
                self.codes.append(mid)
                # leaf level
                for k in range(5):
                    leaf = f"C{i}{j}{k}"
                    self.G.add_node(leaf, depth=2)
                    self.G.add_edge(mid, leaf)
                    self.codes.append(leaf)

        if extra_depth > 0:
            base_leaves = [c for c in self.codes if self.G.out_degree(c) == 0]
            for leaf in base_leaves:
                parent = leaf
                for d in range(extra_depth):
                    child = f"{leaf}d{d}"
                    depth = self.G.nodes[parent]["depth"] + 1
                    self.G.add_node(child, depth=depth)
                    self.G.add_edge(parent, child)
                    self.codes.append(child)
                    parent = child

        self._finalize()

    def _finalize(self):
        self.code2idx = {c: i for i, c in enumerate(self.codes)}
        self.idx2code = {i: c for c, i in self.code2idx.items()}
        depths = nx.get_node_attributes(self.G, "depth")
        self.max_depth = max(depths.values()) if depths else 0
        self.leaf_codes = [c for c in self.codes if self.G.out_degree(c) == 0]

    def depth(self, code: str) -> int:
        return self.G.nodes[code]["depth"]

    def tree_distance(self, c1: str, c2: str) -> int:
        # undirected shortest path; nodes under different roots have no path
        try:
            return nx.shortest_path_length(self.G.to_undirected(), c1, c2)
        except nx.NetworkXNoPath:
            return None


def sample_toy_trajectories(hier: ToyICDHierarchy,
                            num_patients: int = 10000,
                            min_T: int = 3,
                            max_T: int = 6,
                            min_codes_per_visit: int = 2,
                            max_codes_per_visit: int = 5):
    """
    Returns a list of trajectories.
    Each trajectory: list of visits.
    Each visit: list of code indices.
    """
    trajs = []
    all_leaf_codes = hier.leaf_codes

    for _ in range(num_patients):
        T = random.randint(min_T, max_T)
        # choose a random leaf as "disease cluster center"
        base_leaf = random.choice(all_leaf_codes)
        # get its ancestors for comorbid structure
        parents = list(hier.G.predecessors(base_leaf))
        cluster = [base_leaf] + parents

        traj = []
        for _ in range(T):
            num_codes = random.randint(min_codes_per_visit, max_codes_per_visit)
            visit_codes = set()
            # mostly sample from same cluster + some random noise
            while len(visit_codes) < num_codes:
                if random.random() < 0.7:
                    c = random.choice(cluster)
                else:
                    c = random.choice(hier.codes)
                visit_codes.add(hier.code2idx[c])
            traj.append(sorted(list(visit_codes)))
        trajs.append(traj)
    return trajs


def _hierarchy_positions(hier: ToyICDHierarchy):
    depth_groups = {}
    for code in hier.codes:
        d = hier.depth(code)
        depth_groups.setdefault(d, []).append(code)

    positions = {}
    for depth, codes in depth_groups.items():
        codes_sorted = sorted(codes)
        n = len(codes_sorted)
        for idx, code in enumerate(codes_sorted):
            x = (idx + 1) / (n + 1)
            y = -depth
            positions[code] = (x, y)
    return positions


def plot_hierarchy_graph(hier: ToyICDHierarchy, title: str, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pos = _hierarchy_positions(hier)
    plt.figure(figsize=(20, 20))
    nx.draw(
        hier.G,
        pos,
        with_labels=True,
        node_size=200,
        font_size=6,
        arrows=True,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


if __name__ == "__main__":
    hier = ToyICDHierarchy()
    trajs = sample_toy_trajectories(hier)
    print(len(hier.codes), "codes")
    print("Example traj:", trajs[0])

    graphs_dir = os.path.join("plots", "graphs")
    base_hier = ToyICDHierarchy(extra_depth=0)
    extended_hier = ToyICDHierarchy(extra_depth=5)
    plot_hierarchy_graph(
        base_hier,
        "Toy ICD Hierarchy (extra_depth=0)",
        os.path.join(graphs_dir, "hierarchy_depth2.png"),
    )
    plot_hierarchy_graph(
        extended_hier,
        "Toy ICD Hierarchy (extra_depth=5)",
        os.path.join(graphs_dir, "hierarchy_depth7.png"),
    )
