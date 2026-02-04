"""
Bond Visualization - Plotting and visualization for moral bond analysis.

Provides clustering, dimensionality reduction, and visualization functions
for analyzing moral bond collections.
"""

from collections import Counter
from typing import Optional

import numpy as np

from .moral_structure import MoralBond, BondType, RoleType
from .bond_algebra import compute_tradition_similarity, compute_tradition_vectors


# =============================================================================
# CLUSTERING
# =============================================================================


def encode_bonds_as_vectors(bonds: list[MoralBond]) -> tuple[np.ndarray, list[str]]:
    """
    Encode bonds as feature vectors for clustering.

    Each bond is encoded as a one-hot vector over:
    - bond_type (9 values)
    - agent_role (25 values)
    - patient_role (25 values)
    - action (22 values)

    Returns:
        (feature_matrix, feature_names) tuple
    """
    # Build feature indices
    bond_types = [bt.value for bt in BondType]
    roles = [r.value for r in RoleType]
    from .moral_structure import ActionCategory

    actions = [a.value for a in ActionCategory]

    n_features = len(bond_types) + 2 * len(roles) + len(actions)
    feature_names = (
        [f"bt_{bt}" for bt in bond_types]
        + [f"agent_{r}" for r in roles]
        + [f"patient_{r}" for r in roles]
        + [f"action_{a}" for a in actions]
    )

    # Encode bonds
    X = np.zeros((len(bonds), n_features), dtype=np.float32)

    bt_offset = 0
    agent_offset = len(bond_types)
    patient_offset = agent_offset + len(roles)
    action_offset = patient_offset + len(roles)

    bt_map = {bt: i for i, bt in enumerate(bond_types)}
    role_map = {r: i for i, r in enumerate(roles)}
    action_map = {a: i for i, a in enumerate(actions)}

    for i, bond in enumerate(bonds):
        # Bond type
        bt_idx = bt_map.get(bond.bond_type.value, -1)
        if bt_idx >= 0:
            X[i, bt_offset + bt_idx] = 1.0

        # Agent role
        agent_idx = role_map.get(bond.agent_role.value, -1)
        if agent_idx >= 0:
            X[i, agent_offset + agent_idx] = 1.0

        # Patient role
        patient_idx = role_map.get(bond.patient_role.value, -1)
        if patient_idx >= 0:
            X[i, patient_offset + patient_idx] = 1.0

        # Action
        action_idx = action_map.get(bond.action.value, -1)
        if action_idx >= 0:
            X[i, action_offset + action_idx] = 1.0

    return X, feature_names


def cluster_bonds(
    bonds: list[MoralBond],
    n_clusters: int = 5,
    method: str = "kmeans",
) -> dict:
    """
    Cluster bonds based on their feature vectors.

    Args:
        bonds: List of MoralBond objects
        n_clusters: Number of clusters
        method: Clustering method ('kmeans' or 'hierarchical')

    Returns:
        Dict with cluster assignments and statistics
    """
    if len(bonds) < n_clusters:
        return {"error": f"Not enough bonds ({len(bonds)}) for {n_clusters} clusters"}

    X, feature_names = encode_bonds_as_vectors(bonds)

    try:
        from sklearn.cluster import KMeans, AgglomerativeClustering

        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == "hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            return {"error": f"Unknown method: {method}"}

        labels = model.fit_predict(X)

    except ImportError:
        # Fallback to simple hashing-based clustering
        labels = np.array([hash(bond.to_canonical_tuple()) % n_clusters for bond in bonds])

    # Compute cluster statistics
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        cluster_bonds = [b for b, l in zip(bonds, labels) if l == cluster_id]
        if cluster_bonds:
            cluster_stats[cluster_id] = {
                "size": len(cluster_bonds),
                "bond_types": Counter(b.bond_type.value for b in cluster_bonds).most_common(3),
                "agent_roles": Counter(b.agent_role.value for b in cluster_bonds).most_common(3),
                "actions": Counter(b.action.value for b in cluster_bonds).most_common(3),
                "traditions": Counter(
                    b.source_tradition for b in cluster_bonds if b.source_tradition
                ).most_common(3),
            }

    return {
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "method": method,
        "cluster_stats": cluster_stats,
    }


# =============================================================================
# DIMENSIONALITY REDUCTION
# =============================================================================


def reduce_dimensions(
    bonds: list[MoralBond],
    method: str = "pca",
    n_components: int = 2,
) -> tuple[np.ndarray, dict]:
    """
    Reduce bond vectors to 2D/3D for visualization.

    Args:
        bonds: List of MoralBond objects
        method: 'pca', 'tsne', or 'umap'
        n_components: Number of dimensions (2 or 3)

    Returns:
        (coordinates, metadata) tuple
    """
    X, feature_names = encode_bonds_as_vectors(bonds)

    metadata = {
        "method": method,
        "n_components": n_components,
        "n_samples": len(bonds),
    }

    try:
        if method == "pca":
            from sklearn.decomposition import PCA

            model = PCA(n_components=n_components)
            coords = model.fit_transform(X)
            metadata["explained_variance"] = model.explained_variance_ratio_.tolist()

        elif method == "tsne":
            from sklearn.manifold import TSNE

            perplexity = min(30, len(bonds) - 1)
            model = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            coords = model.fit_transform(X)

        elif method == "umap":
            try:
                import umap

                n_neighbors = min(15, len(bonds) - 1)
                model = umap.UMAP(
                    n_components=n_components, n_neighbors=n_neighbors, random_state=42
                )
                coords = model.fit_transform(X)
            except ImportError:
                metadata["error"] = "umap-learn not installed"
                return np.zeros((len(bonds), n_components)), metadata

        else:
            metadata["error"] = f"Unknown method: {method}"
            return np.zeros((len(bonds), n_components)), metadata

    except ImportError as e:
        metadata["error"] = f"sklearn not installed: {e}"
        # Fallback: random projection
        np.random.seed(42)
        projection = np.random.randn(X.shape[1], n_components)
        coords = X @ projection
        metadata["fallback"] = "random_projection"

    return coords, metadata


# =============================================================================
# PLOTTING (text-based for environments without matplotlib)
# =============================================================================


def plot_tradition_heatmap_text(bonds: list[MoralBond]) -> str:
    """
    Generate text-based heatmap of tradition similarities.

    Returns ASCII art representation.
    """
    try:
        similarity, traditions = compute_tradition_similarity(bonds)
    except Exception as e:
        return f"Error computing similarity: {e}"

    if len(traditions) < 2:
        return "Not enough traditions for comparison"

    lines = []
    lines.append("TRADITION SIMILARITY HEATMAP")
    lines.append("=" * 60)

    # Header
    header = "          " + " ".join(f"{t[:8]:>8s}" for t in traditions)
    lines.append(header)

    # Rows
    for i, t1 in enumerate(traditions):
        row = f"{t1[:8]:>8s}  "
        for j, t2 in enumerate(traditions):
            val = similarity[i, j]
            # Use characters to represent values
            if val > 0.8:
                char = "#"
            elif val > 0.6:
                char = "X"
            elif val > 0.4:
                char = "+"
            elif val > 0.2:
                char = "-"
            else:
                char = "."
            row += f"{char * 8} "
        lines.append(row)

    lines.append("")
    lines.append("Legend: # >0.8  X >0.6  + >0.4  - >0.2  . <0.2")

    return "\n".join(lines)


def plot_cluster_summary_text(cluster_result: dict) -> str:
    """Generate text summary of clustering results."""
    lines = []
    lines.append("CLUSTER SUMMARY")
    lines.append("=" * 60)

    if "error" in cluster_result:
        lines.append(f"Error: {cluster_result['error']}")
        return "\n".join(lines)

    lines.append(f"Method: {cluster_result['method']}")
    lines.append(f"Clusters: {cluster_result['n_clusters']}")
    lines.append("")

    for cluster_id, stats in cluster_result.get("cluster_stats", {}).items():
        lines.append(f"Cluster {cluster_id} (n={stats['size']}):")

        if stats["bond_types"]:
            types_str = ", ".join(f"{bt}({c})" for bt, c in stats["bond_types"])
            lines.append(f"  Bond types: {types_str}")

        if stats["agent_roles"]:
            roles_str = ", ".join(f"{r}({c})" for r, c in stats["agent_roles"])
            lines.append(f"  Agent roles: {roles_str}")

        if stats["traditions"]:
            trad_str = ", ".join(f"{t}({c})" for t, c in stats["traditions"])
            lines.append(f"  Traditions: {trad_str}")

        lines.append("")

    return "\n".join(lines)


def plot_bond_space_text(
    bonds: list[MoralBond],
    color_by: str = "tradition",
    width: int = 60,
    height: int = 20,
) -> str:
    """
    Generate text-based scatter plot of bond space.

    Args:
        bonds: List of MoralBond objects
        color_by: 'tradition', 'bond_type', or 'language'
        width: Plot width in characters
        height: Plot height in characters

    Returns:
        ASCII art scatter plot
    """
    if len(bonds) < 2:
        return "Not enough bonds for visualization"

    # Reduce to 2D
    coords, metadata = reduce_dimensions(bonds, method="pca", n_components=2)

    if "error" in metadata:
        return f"Error: {metadata['error']}"

    # Normalize coordinates to grid
    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1

    x_norm = ((x - x_min) / (x_max - x_min) * (width - 1)).astype(int)
    y_norm = ((y - y_min) / (y_max - y_min) * (height - 1)).astype(int)

    # Get colors/symbols based on category
    if color_by == "tradition":
        categories = [b.source_tradition or "unknown" for b in bonds]
    elif color_by == "bond_type":
        categories = [b.bond_type.value for b in bonds]
    elif color_by == "language":
        categories = [b.source_language or "unknown" for b in bonds]
    else:
        categories = ["*"] * len(bonds)

    unique_cats = list(set(categories))
    symbols = "o*x+#@$%&="
    cat_to_symbol = {cat: symbols[i % len(symbols)] for i, cat in enumerate(unique_cats)}

    # Build grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    for i, (xi, yi) in enumerate(zip(x_norm, y_norm)):
        xi = max(0, min(width - 1, xi))
        yi = max(0, min(height - 1, height - 1 - yi))  # Flip y
        symbol = cat_to_symbol[categories[i]]
        grid[yi][xi] = symbol

    # Build output
    lines = []
    lines.append(f"BOND SPACE (colored by {color_by})")
    lines.append("=" * width)

    # Add border and grid
    lines.append("+" + "-" * width + "+")
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * width + "+")

    # Legend
    lines.append("")
    lines.append("Legend:")
    for cat in unique_cats[:10]:  # Limit legend
        lines.append(f"  {cat_to_symbol[cat]} {cat}")

    if metadata.get("explained_variance"):
        ev = metadata["explained_variance"]
        lines.append(f"\nPCA variance: PC1={ev[0]:.1%}, PC2={ev[1]:.1%}")

    return "\n".join(lines)


# =============================================================================
# MATPLOTLIB PLOTTING (optional)
# =============================================================================


def plot_tradition_heatmap(bonds: list[MoralBond], save_path: Optional[str] = None):
    """
    Plot heatmap of tradition similarities using matplotlib.

    Args:
        bonds: List of MoralBond objects
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed. Use plot_tradition_heatmap_text() instead.")
        return

    similarity, traditions = compute_tradition_similarity(bonds)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        similarity,
        xticklabels=traditions,
        yticklabels=traditions,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
    )
    ax.set_title("Tradition Similarity (Cosine)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_bond_space(
    bonds: list[MoralBond],
    color_by: str = "tradition",
    method: str = "pca",
    save_path: Optional[str] = None,
):
    """
    Plot bonds in 2D space using matplotlib.

    Args:
        bonds: List of MoralBond objects
        color_by: 'tradition', 'bond_type', or 'language'
        method: Dimensionality reduction method
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Use plot_bond_space_text() instead.")
        return

    coords, metadata = reduce_dimensions(bonds, method=method, n_components=2)

    if color_by == "tradition":
        colors = [b.source_tradition or "unknown" for b in bonds]
    elif color_by == "bond_type":
        colors = [b.bond_type.value for b in bonds]
    elif color_by == "language":
        colors = [b.source_language or "unknown" for b in bonds]
    else:
        colors = ["bond"] * len(bonds)

    unique_colors = list(set(colors))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    color_indices = [color_map[c] for c in colors]

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color_indices,
        cmap="tab10",
        alpha=0.6,
        s=50,
    )

    # Legend
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=plt.cm.tab10(i / 10), markersize=10
        )
        for i in range(len(unique_colors))
    ]
    ax.legend(handles, unique_colors, loc="best", title=color_by.title())

    ax.set_title(f"Bond Space ({method.upper()}, colored by {color_by})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
