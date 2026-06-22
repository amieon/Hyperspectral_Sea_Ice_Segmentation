import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

BAND_FILES = [f"../data/1-{i}.png" for i in range(1, 6)]
N_CLUSTERS = 4
LATENT_DIM = 2
RANDOM_STATE = 0
EPOCHS = 60
COLOR_MAP = {
    0: (0, 0, 0), 1: (0, 200, 255), 2: (255, 255, 255),
    3: (180, 180, 180), 4: (0, 40, 120),
}

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class Autoencoder(nn.Module):
    """5 -> 8 -> 2 -> 8 -> 5 的对称自编码器, 编码器含非线性激活。"""
    def __init__(self, in_dim=5, latent=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 8), nn.ReLU(),
            nn.Linear(8, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 8), nn.ReLU(),
            nn.Linear(8, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def load_cube(band_files):
    bands = [np.array(Image.open(f).convert("L")).astype(np.float64) for f in band_files]
    return np.stack(bands, axis=-1)


def colorize(label_img):
    H, W = label_img.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for k, c in COLOR_MAP.items():
        rgb[label_img == k] = c
    return rgb


def align_to_pca(pca_ids, ae_ids, k=N_CLUSTERS):
    """把 AE 的簇号对齐到 PCA, 便于配色对比。"""
    from scipy.optimize import linear_sum_assignment
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M[i, j] = np.sum((pca_ids == i) & (ae_ids == j))
    r, c = linear_sum_assignment(-M)
    mapping = {cj: ri for ri, cj in zip(r, c)}
    return np.array([mapping[i] for i in ae_ids])


def main():
    cube = load_cube(BAND_FILES)
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    mask = ~np.all(X == 0, axis=1)
    Xv = X[mask]

    # 标准化(神经网络训练需要)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xv)

    # ---------- 方案 A: PCA + K-means (经典基线) ----------
    pca = PCA(n_components=LATENT_DIM, random_state=RANDOM_STATE)
    Zp = pca.fit_transform(Xs)
    pca_ids = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE).fit_predict(Zp)
    pca_recon = pca.inverse_transform(Zp)
    pca_mse = np.mean((Xs - pca_recon) ** 2)
    pca_sil = silhouette_score(Zp[np.random.RandomState(0).choice(len(Zp), 10000, replace=False)],
                               pca_ids[np.random.RandomState(0).choice(len(Zp), 10000, replace=False)])

    # ---------- 方案 B: Autoencoder + K-means (深度学习) ----------
    Xt = torch.tensor(Xs, dtype=torch.float32)
    model = Autoencoder(in_dim=B, latent=LATENT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    print("=" * 55)
    print("训练自编码器 ...")
    print("=" * 55)
    n = Xt.shape[0]
    batch = 4096
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xt[idx]
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1:3d}/{EPOCHS}  recon MSE = {tot/n:.5f}")

    model.eval()
    with torch.no_grad():
        recon_all, Z_ae = model(Xt)
        Z_ae = Z_ae.numpy()
        ae_mse = float(((recon_all - Xt) ** 2).mean())

    ae_ids = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE).fit_predict(Z_ae)
    sidx = np.random.RandomState(0).choice(len(Z_ae), 10000, replace=False)
    ae_sil = silhouette_score(Z_ae[sidx], ae_ids[sidx])

    ae_ids_aligned = align_to_pca(pca_ids, ae_ids)

    # ---------- 结果对比 ----------
    print("\n" + "=" * 55)
    print("经典 PCA  vs  深度自编码器")
    print("=" * 55)
    print(f"{'方案':<22}{'重建MSE↓':<12}{'轮廓系数↑':<12}")
    print(f"{'PCA + K-means':<22}{pca_mse:<12.5f}{pca_sil:<12.4f}")
    print(f"{'Autoencoder + K-means':<22}{ae_mse:<12.5f}{ae_sil:<12.4f}")

    # 出图
    def to_img(ids):
        full = np.zeros(X.shape[0], dtype=int)
        full[mask] = ids + 1
        return full.reshape(H, W)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].imshow(colorize(to_img(pca_ids)))
    axes[0].set_title(f"PCA + K-means\n(Silhouette={pca_sil:.3f})")
    axes[0].axis("off")
    axes[1].imshow(colorize(to_img(ae_ids_aligned)))
    axes[1].set_title(f"Autoencoder + K-means\n(Silhouette={ae_sil:.3f})")
    axes[1].axis("off")
    plt.suptitle("Classic PCA vs Deep Autoencoder", fontsize=14)
    plt.tight_layout()
    plt.savefig("../figures/ae_vs_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[图] PCA vs 自编码器 对比已保存: ../figures/ae_vs_pca.png")

    # 隐空间散点对比
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    s = np.random.RandomState(0).choice(len(Zp), 5000, replace=False)
    axes[0].scatter(Zp[s, 0], Zp[s, 1], c=pca_ids[s], cmap="tab10", s=4, alpha=0.5)
    axes[0].set_title("PCA latent space"); axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[1].scatter(Z_ae[s, 0], Z_ae[s, 1], c=ae_ids[s], cmap="tab10", s=4, alpha=0.5)
    axes[1].set_title("Autoencoder latent space"); axes[1].set_xlabel("z1"); axes[1].set_ylabel("z2")
    plt.tight_layout()
    plt.savefig("../figures/ae_latent_space.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[图] 隐空间对比已保存: ../figures/ae_latent_space.png")


if __name__ == "__main__":
    main()
