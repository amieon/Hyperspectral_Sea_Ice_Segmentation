import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

BAND_FILES = [f"../data/1-{i}.png" for i in range(1, 6)]
N_CLUSTERS = 4
RANDOM_STATE = 0
SPECTRAL_SAMPLE = 3000   # 谱聚类子采样数量
COLOR_MAP = {
    0: (0, 0, 0), 1: (0, 200, 255), 2: (255, 255, 255),
    3: (180, 180, 180), 4: (0, 40, 120),
}


def load_cube(band_files):
    bands = [np.array(Image.open(f).convert("L")).astype(np.float64) for f in band_files]
    return np.stack(bands, axis=-1)


def colorize(label_img):
    H, W = label_img.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for k, c in COLOR_MAP.items():
        rgb[label_img == k] = c
    return rgb


def order_by_brightness(ids, Xv_raw, k=N_CLUSTERS):
    """按各簇原始光谱均值亮度排序, 让不同方法的簇号语义一致(便于配色对比)。"""
    means = [Xv_raw[ids == c].mean() if (ids == c).any() else 1e9 for c in range(k)]
    order = np.argsort(means)            # 暗->亮
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[i] for i in ids])


def run_kmeans(Xp):
    return KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE).fit_predict(Xp)


def run_gmm(Xp):
    return GaussianMixture(n_components=N_CLUSTERS, covariance_type="full",
                           random_state=RANDOM_STATE).fit_predict(Xp)


def run_spectral(Xp):
    """子采样训练 + KNN 外推。"""
    n = Xp.shape[0]
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(n, size=min(SPECTRAL_SAMPLE, n), replace=False)
    sc = SpectralClustering(n_clusters=N_CLUSTERS, affinity="nearest_neighbors",
                            n_neighbors=10, assign_labels="kmeans",
                            random_state=RANDOM_STATE)
    sub_labels = sc.fit_predict(Xp[idx])
    # 用子样本标签训练 KNN, 外推到全部像素
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xp[idx], sub_labels)
    return knn.predict(Xp)


def evaluate(Xp, ids):
    n = Xp.shape[0]
    sidx = np.random.RandomState(0).choice(n, size=min(10000, n), replace=False)
    return (silhouette_score(Xp[sidx], ids[sidx]),
            calinski_harabasz_score(Xp, ids),
            davies_bouldin_score(Xp, ids))


def main():
    cube = load_cube(BAND_FILES)
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    mask = ~np.all(X == 0, axis=1)
    Xv = X[mask]
    Xv_b1 = Xv[:, 0]                       # 用于按亮度排序
    Xp = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xv)

    methods = {
        "K-means": run_kmeans,
        "GMM": run_gmm,
        "Spectral": run_spectral,
    }

    results, metrics = {}, {}
    for name, fn in methods.items():
        print(f"[运行] {name} ...")
        ids = fn(Xp)
        ids = order_by_brightness(ids, Xv_b1)   # 统一簇号语义
        results[name] = ids
        metrics[name] = evaluate(Xp, ids)

    # 打印指标表
    print("\n" + "=" * 60)
    print("聚类方法对比 (基于 PCA 特征)")
    print("=" * 60)
    print(f"{'方法':<12}{'轮廓系数↑':<12}{'CH指数↑':<14}{'DB指数↓':<10}")
    for name in methods:
        s, ch, db = metrics[name]
        print(f"{name:<12}{s:<12.4f}{ch:<14.1f}{db:<10.4f}")
    best = max(metrics, key=lambda m: metrics[m][0])
    print(f"\n>> 按轮廓系数, 表现最佳: {best}")

    # 并排分割图
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, name in zip(axes, methods):
        full = np.zeros(X.shape[0], dtype=int)
        full[mask] = results[name] + 1
        ax.imshow(colorize(full.reshape(H, W)))
        s = metrics[name][0]
        ax.set_title(f"{name}\n(Silhouette={s:.3f})", fontsize=12)
        ax.axis("off")
    plt.suptitle("Clustering method comparison", fontsize=15)
    plt.tight_layout()
    plt.savefig("../figures/cluster_methods_compare.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[图] 方法对比分割图已保存: ../figures/cluster_methods_compare.png")

    # 指标柱状图
    names = list(methods.keys())
    sil = [metrics[n][0] for n in names]
    db = [metrics[n][2] for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(names, sil, color=["#3498db", "#e67e22", "#2ecc71"])
    ax1.set_title("Silhouette (higher better)"); ax1.set_ylim(0, 1)
    for i, v in enumerate(sil): ax1.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax2.bar(names, db, color=["#3498db", "#e67e22", "#2ecc71"])
    ax2.set_title("Davies-Bouldin (lower better)")
    for i, v in enumerate(db): ax2.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig("../figures/cluster_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[图] 指标对比图已保存: cluster_metrics.png")


if __name__ == "__main__":
    main()
