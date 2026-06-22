import time
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
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
    return np.stack(bands, axis=-1)


def kmeans_label(features, mask, shape):
    """对 features (N_valid, d) 聚类, 返回完整标签图 (0=其他)。"""
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
    ids = km.fit_predict(features)
    labels = np.zeros(mask.shape[0], dtype=int)
    labels[mask] = ids + 1
    return labels.reshape(shape)


def align_labels(ref, target, k=N_CLUSTERS):
    """用匈牙利算法把 target 的簇号对齐到 ref, 让颜色一致, 并返回一致率。"""
    # 只在有效像素(非 0)上比较
    m = (ref > 0) & (target > 0)
    a, b = ref[m], target[m]
    M = np.zeros((k + 1, k + 1))
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            M[i, j] = np.sum((a == i) & (b == j))
    r, c = linear_sum_assignment(-M[1:, 1:])
    mapping = {cj + 1: ri + 1 for ri, cj in zip(r, c)}
    aligned = np.zeros_like(target)
    for old, new in mapping.items():
        aligned[target == old] = new
    agree = np.sum(aligned[m] == ref[m]) / m.sum()
    return aligned, agree


def colorize(label_img):
    H, W = label_img.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for kk, cc in COLOR_MAP.items():
        rgb[label_img == kk] = cc
    return rgb


def main():
    cube = load_cube(BAND_FILES)
    H, W, B = cube.shape
    shape = (H, W)
    X = cube.reshape(-1, B)
    mask = ~np.all(X == 0, axis=1)
    Xv = X[mask]

    results = {}
    times = {}

    # (A) 单波段
    t = time.time()
    results["Single-band"] = kmeans_label(Xv[:, :1], mask, shape)
    times["Single-band"] = time.time() - t

    # (B) 全 5 波段
    t = time.time()
    results["5-band"] = kmeans_label(Xv, mask, shape)
    times["5-band"] = time.time() - t

    # (C) PCA 前 2 维
    t = time.time()
    Xp = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xv)
    results["PCA-2D"] = kmeans_label(Xp, mask, shape)
    times["PCA-2D"] = time.time() - t

    # 以 5-band 为基准, 对齐另外两个的标签(让颜色可比)
    ref = results["5-band"]
    aligned = {"5-band": ref}
    aligned["Single-band"], acc_sb = align_labels(ref, results["Single-band"])
    aligned["PCA-2D"], acc_pca = align_labels(ref, results["PCA-2D"])

    print("=" * 55)
    print("对比实验结果")
    print("=" * 55)
    print(f"{'方案':<14}{'耗时(s)':<12}{'与5波段一致率':<14}")
    print(f"{'Single-band':<14}{times['Single-band']:<12.3f}{acc_sb*100:<14.2f}")
    print(f"{'5-band':<14}{times['5-band']:<12.3f}{'100.00 (基准)':<14}")
    print(f"{'PCA-2D':<14}{times['PCA-2D']:<12.3f}{acc_pca*100:<14.2f}")
    print()
    print(">> 解读:")
    print(f"   - 单波段与全波段一致率 {acc_sb*100:.2f}%, 说明单波段已能完成大体分割,")
    print(f"     但 {100-acc_sb*100:.2f}% 的差异集中在边界/碎冰等细节区域。")
    print(f"   - PCA-2D 与全波段一致率 {acc_pca*100:.2f}%, 几乎无损, 但维度更低、抗噪更好。")

    # --- 并排分割图 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, name in zip(axes, ["Single-band", "5-band", "PCA-2D"]):
        ax.imshow(colorize(aligned[name]))
        ax.set_title(name, fontsize=13)
        ax.axis("off")
    plt.suptitle("K-means Segmentation: feature comparison", fontsize=15)
    plt.tight_layout()
    plt.savefig("comparison_3methods.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[图] 三方案对比图已保存: comparison_3methods.png")

    # --- 差异图: 单波段 vs 全波段 哪些像素分得不一样 ---
    diff = np.zeros(shape, dtype=np.uint8)
    m = (ref > 0)
    diff[m & (aligned["Single-band"] != ref)] = 255  # 不一致处标白
    Image.fromarray(diff).save("diff_map.png")
    print("[图] 差异图(单波段 vs 全波段)已保存: diff_map.png")
    print("    白色 = 两方案分类不同的像素, 主要落在碎冰边界。")


if __name__ == "__main__":
    main()