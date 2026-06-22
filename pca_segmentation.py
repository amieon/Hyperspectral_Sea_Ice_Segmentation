import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

BAND_FILES = [f"1-{i}.png" for i in range(1, 6)]
N_CLUSTERS = 4
RANDOM_STATE = 0
COLOR_MAP = {
    0: (0,   0,   0),
    1: (0,   200, 255),
    2: (255, 255, 255),
    3: (180, 180, 180),
    4: (0,   40,  120),
}


def load_cube(band_files):
    bands = [np.array(Image.open(f).convert("L")).astype(np.float64) for f in band_files]
    return np.stack(bands, axis=-1)   # (H, W, 5)


def run_pca(Xv, n_components=5):
    """对有效像素做 PCA, 返回 pca 对象和降维结果。"""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(Xv)        # (N, n_components)
    ratio = pca.explained_variance_ratio_

    print("=" * 50)
    print("PCA 主成分分析")
    print("=" * 50)
    print("各主成分方差解释率:")
    cum = 0
    for i, r in enumerate(ratio):
        cum += r
        print(f"    PC{i+1}: {r*100:6.3f}%   (累计 {cum*100:6.3f}%)")
    print(f"\n>> 第一主成分(PC1)单独解释了 {ratio[0]*100:.2f}% 的信息。")
    n_keep = int(np.searchsorted(np.cumsum(ratio), 0.99) + 1)
    print(f">> 保留 {n_keep} 个主成分即可覆盖 99% 以上信息 (5 维 -> {n_keep} 维)。")
    return pca, Xp, ratio


def plot_variance(ratio, out="pca_variance.png"):
    n = len(ratio)
    cum = np.cumsum(ratio)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, n + 1), ratio * 100, alpha=0.6, label="Individual")
    ax.plot(range(1, n + 1), cum * 100, "ro-", label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Explained Variance")
    ax.set_xticks(range(1, n + 1))
    ax.legend()
    for i, r in enumerate(ratio):
        ax.text(i + 1, r * 100 + 2, f"{r*100:.2f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[图] 方差解释率图已保存: {out}")


def plot_pca_scatter(Xp, labels_valid, out="pca_scatter.png"):
    """像素在 PC1-PC2 平面的分布, 按聚类结果上色。"""
    idx = np.random.RandomState(0).choice(Xp.shape[0], size=min(5000, Xp.shape[0]), replace=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(Xp[idx, 0], Xp[idx, 1], c=labels_valid[idx], cmap="tab10", s=4, alpha=0.5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Pixels in PC1-PC2 space (colored by cluster)")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图] PCA 散点图已保存: {out}")


def colorize(label_img, color_map=COLOR_MAP):
    H, W = label_img.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for k, c in color_map.items():
        rgb[label_img == k] = c
    return rgb


def main():
    cube = load_cube(BAND_FILES)
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    mask = ~np.all(X == 0, axis=1)
    Xv = X[mask]

    # --- PCA ---
    pca, Xp, ratio = run_pca(Xv, n_components=B)
    plot_variance(ratio)

    # --- 用前 2 个主成分做聚类 (够覆盖 99%+) ---
    n_keep = 2
    Xp_keep = Xp[:, :n_keep]
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
    cluster_ids = km.fit_predict(Xp_keep)

    plot_pca_scatter(Xp, cluster_ids)

    # --- 组装分割图 ---
    labels = np.zeros(X.shape[0], dtype=int)
    labels[mask] = cluster_ids + 1
    label_img = labels.reshape(H, W)
    rgb = colorize(label_img)
    Image.fromarray(rgb).save("seg_pca.png")
    print(f"\n[图] PCA 分割结果已保存: seg_pca.png")
    print(f">> 仅用前 {n_keep} 个主成分 (覆盖 {np.cumsum(ratio)[:n_keep][-1]*100:.2f}% 信息) 完成分割。")


if __name__ == "__main__":
    main()