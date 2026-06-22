import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import ndimage
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


def colorize(label_img):
    H, W = label_img.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for kk, cc in COLOR_MAP.items():
        rgb[label_img == kk] = cc
    return rgb


def evaluate(features, ids):
    """计算三种无监督内部评价指标。"""
    # 轮廓系数计算量大, 抽样 10000 点估计
    n = features.shape[0]
    idx = np.random.RandomState(0).choice(n, size=min(10000, n), replace=False)
    sil = silhouette_score(features[idx], ids[idx])
    ch = calinski_harabasz_score(features, ids)
    db = davies_bouldin_score(features, ids)
    return sil, ch, db


def count_fragments(label_img, n_clusters=N_CLUSTERS):
    """统计各类别的连通区域(碎片)总数, 越少说明区域越连贯。"""
    total = 0
    for c in range(1, n_clusters + 1):
        _, num = ndimage.label(label_img == c)
        total += num
    return total


def morphological_clean(label_img, n_clusters=N_CLUSTERS):
    """逐类别做开运算(去孤立点)+闭运算(填小洞), 合成干净标签图。"""
    cleaned = np.zeros_like(label_img)
    struct = ndimage.generate_binary_structure(2, 2)  # 8-邻域
    for c in range(1, n_clusters + 1):
        binary = (label_img == c)
        binary = ndimage.binary_opening(binary, structure=struct, iterations=1)
        binary = ndimage.binary_closing(binary, structure=struct, iterations=1)
        cleaned[binary] = c
    # 后处理可能产生空隙(0), 用最近邻填回(只在有效区域内)
    valid = (label_img > 0)
    holes = valid & (cleaned == 0)
    if holes.any():
        # 用原始标签填补被开运算清掉的有效像素
        cleaned[holes] = label_img[holes]
    return cleaned


def main():
    cube = load_cube(BAND_FILES)
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    mask = ~np.all(X == 0, axis=1)
    Xv = X[mask]

    # 采用模块3的最优方案: PCA 前2维 + K-means
    Xp = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xv)
    ids = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE).fit_predict(Xp)

    labels = np.zeros(X.shape[0], dtype=int)
    labels[mask] = ids + 1
    label_img = labels.reshape(H, W)

    # ---------- 1) 无监督评估 ----------
    sil, ch, db = evaluate(Xp, ids)
    print("=" * 55)
    print("无监督聚类评估 (基于 PCA 特征)")
    print("=" * 55)
    print(f"  轮廓系数 Silhouette      : {sil:.4f}   (越接近1越好)")
    print(f"  CH 指数 Calinski-Harabasz: {ch:.1f}   (越大越好)")
    print(f"  DB 指数 Davies-Bouldin   : {db:.4f}   (越小越好)")
    if sil > 0.5:
        print("  >> 轮廓系数 > 0.5, 聚类结构清晰, 4 类区分良好。")

    # ---------- 2) 形态学后处理 ----------
    frag_before = count_fragments(label_img)
    cleaned = morphological_clean(label_img)
    frag_after = count_fragments(cleaned)
    print("\n" + "=" * 55)
    print("形态学后处理 (开运算去噪 + 闭运算填洞)")
    print("=" * 55)
    print(f"  后处理前 连通碎片总数: {frag_before}")
    print(f"  后处理后 连通碎片总数: {frag_after}")
    print(f"  碎片减少: {frag_before - frag_after} "
          f"({(1 - frag_after/frag_before)*100:.1f}% 的椒盐噪声被清除)")

    # ---------- 3) 出图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].imshow(colorize(label_img)); axes[0].set_title("Before (raw K-means)"); axes[0].axis("off")
    axes[1].imshow(colorize(cleaned));   axes[1].set_title("After (morphological)"); axes[1].axis("off")
    plt.suptitle("Post-processing comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("postprocess_compare.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[图] 后处理前后对比已保存: postprocess_compare.png")

    Image.fromarray(colorize(cleaned)).save("final_segmentation.png")
    print("[图] 最终分割结果已保存: final_segmentation.png")


if __name__ == "__main__":
    main()
