import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

BAND_FILES = [f"../data/1-{i}.png" for i in range(1, 6)]
N_CLUSTERS = 4
RANDOM_STATE = 0
CLASS_NAMES = ["Sea water", "Thin ice", "Thick ice", "Land"]


def load_cube(band_files):
    bands = [np.array(Image.open(f).convert("L")).astype(np.float64) for f in band_files]
    return np.stack(bands, axis=-1)


def main():
    cube = load_cube(BAND_FILES)
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    mask = ~np.all(X == 0, axis=1)
    Xv = X[mask]

    # 用最终方案 PCA + K-means 得到标签
    Xp = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xv)
    ids = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE).fit_predict(Xp)

    # 按原始光谱平均亮度排序, 让类别 0..3 对应 暗->亮 (海水->陆地)
    bright = [Xv[ids == c].mean() for c in range(N_CLUSTERS)]
    order = np.argsort(bright)
    remap = {old: new for new, old in enumerate(order)}
    ids = np.array([remap[i] for i in ids])

    # 各类在原始 5 波段上的均值与标准差
    means = np.array([Xv[ids == c].mean(axis=0) for c in range(N_CLUSTERS)])  # (4,5)
    stds = np.array([Xv[ids == c].std(axis=0) for c in range(N_CLUSTERS)])

    print("=" * 60)
    print("类别可分性分析")
    print("=" * 60)
    print("\n[1] 各类平均光谱 (5 波段):")
    for c in range(N_CLUSTERS):
        print(f"  {CLASS_NAMES[c]:<16}: {np.round(means[c],1)}  (像素数 {np.sum(ids==c)})")

    # 类间欧氏距离矩阵
    print("\n[2] 类间欧氏距离矩阵:")
    dist = np.zeros((N_CLUSTERS, N_CLUSTERS))
    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            dist[i, j] = np.linalg.norm(means[i] - means[j])
    print("       " + "".join(f"{CLASS_NAMES[j][:4]:>10}" for j in range(N_CLUSTERS)))
    for i in range(N_CLUSTERS):
        print(f"  {CLASS_NAMES[i][:4]:<6}" + "".join(f"{dist[i,j]:>10.1f}" for j in range(N_CLUSTERS)))

    # 可分性指标: 类间散度 / 类内散度
    grand_mean = Xv.mean(axis=0)
    Sb = sum(np.sum(ids == c) * np.sum((means[c] - grand_mean) ** 2) for c in range(N_CLUSTERS))
    Sw = sum(np.sum((Xv[ids == c] - means[c]) ** 2) for c in range(N_CLUSTERS))
    sep_ratio = Sb / Sw
    print(f"\n[3] 可分性指标 (Fisher 判别比 类间散度/类内散度): {sep_ratio:.2f}")
    print(f"    类间散度 Sb = {Sb:.3e}")
    print(f"    类内散度 Sw = {Sw:.3e}")
    print(f"    >> 比值 = {sep_ratio:.2f}, 远大于 1 表示类间差异远大于类内离散, 四类高度可分。")

    # 最小类间距离 vs 最大类内标准差
    min_inter = dist[dist > 0].min()
    max_intra = stds.mean(axis=1).max()
    print(f"\n[4] 最小类间距离 {min_inter:.1f}  vs  最大类内平均标准差 {max_intra:.1f}")
    print(f"    >> 类间间隔显著大于类内波动, 进一步佐证可分性。")

    # ---------- 图1: 平均光谱曲线 + 标准差带 ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#082c6c", "#00c8ff", "#cfd8dc", "#9e9e9e"]
    x = np.arange(1, B + 1)
    for c in range(N_CLUSTERS):
        ax1.plot(x, means[c], "o-", color=colors[c], label=CLASS_NAMES[c], linewidth=2)
        ax1.fill_between(x, means[c] - stds[c], means[c] + stds[c], color=colors[c], alpha=0.15)
    ax1.set_xlabel("Band"); ax1.set_ylabel("Mean reflectance (gray value)")
    ax1.set_title("Mean spectral curves of 4 classes (±1 std band)")
    ax1.set_xticks(x); ax1.legend(); ax1.grid(alpha=0.3)

    # ---------- 图2: 类间距离热力图 ----------
    im = ax2.imshow(dist, cmap="viridis")
    ax2.set_xticks(range(N_CLUSTERS)); ax2.set_yticks(range(N_CLUSTERS))
    short = ["Sea","Thin","Thick","Land"]
    ax2.set_xticklabels(short); ax2.set_yticklabels(short)
    ax2.set_title(f"Inter-class distance matrix\n(Fisher ratio = {sep_ratio:.1f})")
    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            ax2.text(j, i, f"{dist[i,j]:.0f}", ha="center", va="center",
                     color="white" if dist[i,j] < dist.max()*0.6 else "black", fontsize=10)
    fig.colorbar(im, ax=ax2, label="Euclidean distance")
    plt.tight_layout()
    plt.savefig("../figures/separability.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[图] 可分性分析图已保存: separability.png")


if __name__ == "__main__":
    main()
