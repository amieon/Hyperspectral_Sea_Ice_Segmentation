import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

BAND_FILES = [f"1-{i}.png" for i in range(1, 6)]


def load_valid_pixels(band_files):
    """读取 5 波段, 返回有效像素的 (N, 5) 矩阵 (去掉全 0 的黑边)。"""
    bands = [np.array(Image.open(f).convert("L")).astype(np.float64) for f in band_files]
    X = np.stack([b.ravel() for b in bands], axis=1)   # (H*W, 5)
    mask = ~np.all(X == 0, axis=1)                     # 去掉"其他"
    return X[mask]


def correlation_analysis(Xv):
    """计算并打印相关系数矩阵, 返回它。"""
    corr = np.corrcoef(Xv.T)   # 5x5
    np.set_printoptions(precision=3, suppress=True)

    print("=" * 50)
    print("波段相关性分析")
    print("=" * 50)
    print(f"有效像素数: {Xv.shape[0]}")
    print("\n[1] 各波段统计:")
    for i in range(Xv.shape[1]):
        print(f"    Band{i+1}: 均值={Xv[:,i].mean():6.2f}  标准差={Xv[:,i].std():5.2f}")

    print("\n[2] 5x5 相关系数矩阵:")
    print(corr)

    # 取上三角(不含对角线)的相关系数, 衡量整体冗余
    iu = np.triu_indices(corr.shape[0], k=1)
    off_diag = corr[iu]
    print(f"\n[3] 冗余度结论:")
    print(f"    非对角相关系数范围: {off_diag.min():.3f} ~ {off_diag.max():.3f}")
    print(f"    平均相关系数:       {off_diag.mean():.3f}")
    if off_diag.min() > 0.95:
        print("    >> 所有波段相关系数均 > 0.95, 存在严重信息冗余。")
        print("    >> 结论: 应引入 PCA 降维去除冗余 (见模块 3)。")
    return corr


def plot_heatmap(corr, out="correlation_heatmap.png"):
    """画相关系数热力图。"""
    n = corr.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="YlOrRd", vmin=0.9, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"B{i+1}" for i in range(n)])
    ax.set_yticklabels([f"B{i+1}" for i in range(n)])
    ax.set_title("Band Correlation Matrix")
    # 在每个格子标数值
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i,j]:.3f}", ha="center", va="center",
                    color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Correlation coefficient")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[4] 热力图已保存: {out}")


def plot_scatter(Xv, out="band_scatter.png"):
    """画 Band1 vs 其余波段的散点图, 直观展示'点都挤在对角线上 = 冗余'。"""
    n = Xv.shape[1]
    # 抽样 3000 个点画散点(全画太密)
    idx = np.random.RandomState(0).choice(Xv.shape[0], size=min(3000, Xv.shape[0]), replace=False)
    Xs = Xv[idx]
    fig, axes = plt.subplots(1, n - 1, figsize=(4 * (n - 1), 4))
    for k in range(1, n):
        ax = axes[k - 1]
        ax.scatter(Xs[:, 0], Xs[:, k], s=3, alpha=0.3)
        ax.plot([0, 255], [0, 255], "r--", lw=1)   # y=x 参考线
        ax.set_xlabel("Band 1"); ax.set_ylabel(f"Band {k+1}")
        ax.set_title(f"B1 vs B{k+1}")
    plt.suptitle("Pairwise scatter (points hug y=x line => redundancy)")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[5] 散点图已保存: {out}")


def main():
    Xv = load_valid_pixels(BAND_FILES)
    corr = correlation_analysis(Xv)
    plot_heatmap(corr)
    plot_scatter(Xv)


if __name__ == "__main__":
    main()
