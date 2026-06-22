# -*- coding: utf-8 -*-
"""
高光谱影像分割 —— 格陵兰岛巴芬湾海域海冰分割
方法: K-means 无监督聚类 (逐像素分类)

输入: 5 张 350x300 的单波段灰度图 (1-1.png ~ 1-5.png)
输出: 一张彩色分割图, 把像素分为 4 类 (海水/薄冰/厚冰/陆地) + 其他(无用区)

运行: python main.py
依赖: numpy, pillow, scikit-learn, matplotlib
      pip install numpy pillow scikit-learn matplotlib
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
BAND_FILES = [f"1-{i}.png" for i in range(1, 6)]  # 5 个波段
N_CLUSTERS = 4          # 4 个有效类别: 海水/薄冰/厚冰/陆地
RANDOM_STATE = 0        # 固定随机种子, 保证结果可复现

# 每个类别对应的颜色 (RGB)。0 号是"其他"(无用区), 固定为黑色。
# 1~4 号是 K-means 分出的 4 类, 颜色可自己调。
COLOR_MAP = {
    0: (0,   0,   0),     # 其他 (像素值全为 0 的黑边)
    1: (0,   200, 255),   # 类别1
    2: (255, 255, 255),   # 类别2
    3: (180, 180, 180),   # 类别3
    4: (0,   40,  120),   # 类别4
}


def load_cube(band_files):
    """读取 5 个单波段图, 堆叠成 (H, W, 5) 的高光谱立方体。"""
    bands = []
    for f in band_files:
        img = np.array(Image.open(f).convert("L")).astype(np.float64)
        bands.append(img)
    cube = np.stack(bands, axis=-1)   # (H, W, B)
    print(f"[1] 已读取 {len(band_files)} 个波段, 立方体形状 = {cube.shape}")
    return cube


def segment(cube, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
    """对立方体做逐像素 K-means 聚类。返回标签图 (H, W), 0 表示'其他'。"""
    H, W, B = cube.shape
    X = cube.reshape(-1, B)            # (H*W, B): 每行是一个像素的 5 维光谱向量

    # "其他" = 5 个波段全为 0 的像素 (影像旋转留下的黑边)
    mask_valid = ~np.all(X == 0, axis=1)
    print(f"[2] 有效像素 {mask_valid.sum()} / {X.shape[0]} "
          f"(其余 {np.size(mask_valid) - mask_valid.sum()} 个为'其他')")

    # 只对有效像素聚类
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    cluster_ids = km.fit_predict(X[mask_valid])   # 0..n_clusters-1

    # 组装标签图: 0 留给"其他", 有效像素的标签 +1 变成 1..n_clusters
    labels = np.zeros(X.shape[0], dtype=int)
    labels[mask_valid] = cluster_ids + 1
    label_img = labels.reshape(H, W)

    # 打印每一类的平均光谱, 方便你判断哪个簇是海水/冰/陆地
    print("[3] 各类别平均光谱 (5 个波段的均值):")
    for c in range(n_clusters + 1):
        n = (labels == c).sum()
        if n == 0:
            continue
        spec = X[labels == c].mean(axis=0)
        name = "其他" if c == 0 else f"类别{c}"
        print(f"    {name:>5}: {n:6d} 像素, 光谱 = {np.round(spec, 1)}")

    return label_img


def colorize(label_img, color_map=COLOR_MAP):
    """把标签图上色成 RGB 图。"""
    H, W = label_img.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for k, color in color_map.items():
        rgb[label_img == k] = color
    return rgb


def main():
    cube = load_cube(BAND_FILES)
    label_img = segment(cube)
    rgb = colorize(label_img)

    # 保存结果
    Image.fromarray(rgb).save("segmentation_result.png")
    print("[4] 分割结果已保存到 segmentation_result.png")

    # 可视化对比: 左边原始(第1波段), 右边分割结果
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].imshow(cube[:, :, 0], cmap="gray")
    axes[0].set_title("Band 1 (raw)")
    axes[0].axis("off")
    axes[1].imshow(rgb)
    axes[1].set_title("K-means Segmentation")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
    print("[5] 对比图已保存到 comparison.png")


if __name__ == "__main__":
    main()