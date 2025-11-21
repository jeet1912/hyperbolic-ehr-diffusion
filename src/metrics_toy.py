import numpy as np
from collections import Counter

def traj_stats(trajs, hier):
    """
    trajs: list of trajectories
           each trajectory: list of visits
           each visit: list of code indices
    hier: ToyICDHierarchy
    """
    depths = []
    tree_dists = []
    root_prefix_matches = []  # how often codes in a visit share same root C0..C4

    for traj in trajs:
        for visit in traj:
            # filter possible pad visits
            visit = [c for c in visit if c != -1]
            if len(visit) < 2:
                continue

            codes = [hier.idx2code[i] for i in visit]
            # depth stats
            d = [hier.depth(c) for c in codes]
            depths.extend(d)

            # tree distance stats
            for i in range(len(codes)):
                for j in range(i + 1, len(codes)):
                    dist = hier.tree_distance(codes[i], codes[j])
                    if dist is not None:
                        tree_dists.append(dist)

            # root-prefix purity: fraction of codes sharing majority root C0..C4
            roots = [c[0:2] for c in codes]  # "C0", "C1", ...
            counts = Counter(roots)
            majority_root, majority_count = counts.most_common(1)[0]
            root_prefix_matches.append(majority_count / len(roots))

    return {
        "mean_depth": float(np.mean(depths)),
        "std_depth": float(np.std(depths)),
        "mean_tree_dist": float(np.mean(tree_dists)),
        "std_tree_dist": float(np.std(tree_dists)),
        "mean_root_purity": float(np.mean(root_prefix_matches)),
        "std_root_purity": float(np.std(root_prefix_matches)),
    }
