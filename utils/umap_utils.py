from matplotlib import pyplot as plt


def plot_all_umap(s_2d, fd_df, X, biome_colors, feature_list,):

    plot_umap_by_features(s_2d, X, features=feature_list, name="figures/biome_umap_shap_biotic.png")

    features_list = [feat for feat in X.columns if feat not in features_list]

    plot_umap_by_features(s_2d, X, features=feature_list, name="figures/biome_umap_shap_abiotic.png")


def plot_umap_by_features(s_2d, X, features, cmap='viridis', name=None):
    ncols = 3
    nrows = (len(features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten()
    fig.subplots_adjust(hspace=0.5, wspace=0.4)

    for ax, feat in zip(axes_flat, features):
        sc = ax.scatter(
            s_2d[:, 0], s_2d[:, 1],
            c=X[feat].values,
            cmap=cmap,
            s=5,
            alpha=0.6
        )
        plt.colorbar(sc, ax=ax, label=feat)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.spines[['right', 'top']].set_visible(False)

    # Hide unused axes
    for ax in axes_flat[len(features):]:
        ax.set_visible(False)

    plt.suptitle("UMAP of SHAP values coloured by feature", fontsize=13, y=1.02)
    if name:
        plt.savefig(name, dpi=150, bbox_inches="tight")
    plt.show()


def plot_umap_biome(s_2d, fd_df, biome_colors, name="figures/umap_shap_biome.png"):
    """
    Plots a 2D UMAP embedding of SHAP values colored by biome.

    Parameters:
    - s_2d: 2D numpy array of UMAP embeddings.
    - fd_df: DataFrame containing the original data with a 'biome' column.
    - biome_colors: Dictionary mapping biome names to colors.
    """

    fig, ax = plt.subplots(figsize=(10, 7))
    biomes =  fd_df['biome'].values  # numpy array of biome labels per row
    for biome, color in biome_colors.items():
        mask = biomes == biome
        if mask.sum() == 0:
            continue
        ax.scatter(
            s_2d[mask, 0], s_2d[mask, 1],
            color=color,
            s=5,
            alpha=0.8,
            label=biome
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of SHAP values coloured by Biome")
    ax.legend(
        markerscale=3,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=False,
        fontsize=8
    )
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches="tight")
