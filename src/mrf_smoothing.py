import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import ndimage
import matplotlib.pyplot as plt

BAND_FILES = [f"../data/1-{i}.png" for i in range(1, 6)]
N_CLUSTERS = 4
RANDOM_STATE = 0
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


def count_fragments(label_img, k=N_CLUSTERS):
    total = 0
    for c in range(1, k + 1):
        _, n = ndimage.label(label_img == c)
        total += n
    return total


def mrf_icm(label_img, feat_img, centers, mask2d, beta=1.0, n_iter=8):
    H, W = label_img.shape
    K = centers.shape[0]
    labels = label_img.copy()

    data_cost = np.zeros((H, W, K))
    for k in range(K):
        diff = feat_img - centers[k]
        data_cost[:, :, k] = np.sum(diff ** 2, axis=-1)
    # 标准化数据项: 除以有效像素数据项的均值, 使其量纲稳定在 O(1)。
    # 这样 beta(乘在 0..4 的平滑项上)能真正调节"数据 vs 平滑"的相对权重。
    dmean = data_cost[mask2d].mean() if mask2d.any() else 1.0
    data_cost = data_cost / (dmean + 1e-9)

    for it in range(n_iter):
        new_labels = labels.copy()
        best_cost = None
        best_k = None
        for k in range(K):
            same = (labels == (k + 1)).astype(float)
            neigh_same = (np.roll(same, 1, 0) + np.roll(same, -1, 0) +
                          np.roll(same, 1, 1) + np.roll(same, -1, 1))
            smooth_cost_k = 4.0 - neigh_same
            total_cost_k = data_cost[:, :, k] + beta * smooth_cost_k
            if k == 0:
                best_cost = total_cost_k.copy()
                best_k = np.zeros((H, W), dtype=int)
            else:
                better = total_cost_k < best_cost
                best_cost[better] = total_cost_k[better]
                best_k[better] = k
        cand = best_k + 1
        upd = mask2d & (cand != labels)
        changed = int(upd.sum())
        new_labels[upd] = cand[upd]
        labels = new_labels
        print(f"    ICM iter {it+1}: {changed} 个像素更新")
        if changed == 0:
            break
    return labels


def morph_clean(label_img, k=N_CLUSTERS):
    cleaned = np.zeros_like(label_img)
    st = ndimage.generate_binary_structure(2, 2)
    for c in range(1, k + 1):
        b = (label_img == c)
        b = ndimage.binary_opening(b, st, 1)
        b = ndimage.binary_closing(b, st, 1)
        cleaned[b] = c
    valid = label_img > 0
    holes = valid & (cleaned == 0)
    cleaned[holes] = label_img[holes]
    return cleaned


def main():
    cube = load_cube(BAND_FILES)
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    mask = ~np.all(X == 0, axis=1)
    mask2d = mask.reshape(H, W)
    Xv = X[mask]

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(Xv)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE).fit(Xp)
    ids = km.labels_
    centers = km.cluster_centers_

    labels0 = np.zeros(X.shape[0], dtype=int)
    labels0[mask] = ids + 1
    labels0 = labels0.reshape(H, W)

    feat_full = np.zeros((H * W, 2))
    feat_full[mask] = Xp
    feat_img = feat_full.reshape(H, W, 2)

    frag0 = count_fragments(labels0)
    print("=" * 55)
    print("MRF 平滑 (ICM 求解)")
    print("=" * 55)
    print(f"初始 (raw K-means) 碎片数: {frag0}\n")

    betas = [0.2, 1.0, 3.0]
    mrf_results = {}
    for beta in betas:
        print(f"[beta={beta}]")
        lab = mrf_icm(labels0, feat_img, centers, mask2d, beta=beta, n_iter=8)
        frag = count_fragments(lab)
        mrf_results[beta] = (lab, frag)
        print(f"    -> 碎片数: {frag0} -> {frag} (减少 {(1-frag/frag0)*100:.1f}%)\n")

    morph = morph_clean(labels0)
    frag_m = count_fragments(morph)

    fig, axes = plt.subplots(1, len(betas) + 1, figsize=(5 * (len(betas) + 1), 6))
    axes[0].imshow(colorize(labels0)); axes[0].set_title(f"Raw K-means\n(frag={frag0})"); axes[0].axis("off")
    for ax, beta in zip(axes[1:], betas):
        lab, frag = mrf_results[beta]
        ax.imshow(colorize(lab)); ax.set_title(f"MRF beta={beta}\n(frag={frag})"); ax.axis("off")
    plt.suptitle("MRF smoothing at different smoothness strength", fontsize=14)
    plt.tight_layout(); plt.savefig("../figures/mrf_lambda_compare.png", dpi=150, bbox_inches="tight"); plt.close()
    print("[图] 不同平滑强度对比已保存: ../figures/mrf_lambda_compare.png")

    best_lab, best_frag = mrf_results[1.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(colorize(labels0)); axes[0].set_title(f"Raw\n(frag={frag0})"); axes[0].axis("off")
    axes[1].imshow(colorize(morph));   axes[1].set_title(f"Morphological\n(frag={frag_m})"); axes[1].axis("off")
    axes[2].imshow(colorize(best_lab));axes[2].set_title(f"MRF (beta=1)\n(frag={best_frag})"); axes[2].axis("off")
    plt.suptitle("Post-processing: Morphological vs MRF", fontsize=14)
    plt.tight_layout(); plt.savefig("../figures/mrf_vs_morph.png", dpi=150, bbox_inches="tight"); plt.close()
    print("[图] MRF vs 形态学 对比已保存: ../figures/mrf_vs_morph.png")

    Image.fromarray(colorize(best_lab)).save("../figures/final_mrf.png")
    print("[图] MRF 最终分割已保存: ../figures/final_mrf.png")

    print("\n" + "=" * 55)
    print(f"{'方法':<20}{'碎片数':<10}{'相对raw':<10}")
    print(f"{'Raw K-means':<20}{frag0:<10}{'-':<10}")
    print(f"{'形态学':<18}{frag_m:<10}{f'-{(1-frag_m/frag0)*100:.1f}%':<10}")
    print(f"{'MRF (beta=1)':<20}{best_frag:<10}{f'-{(1-best_frag/frag0)*100:.1f}%':<10}")


if __name__ == "__main__":
    main()
