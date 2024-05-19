from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, label

class RCIC():
    _SCALES = [2 ** i for i in range(5)]
    _ORIENTATIONS = [np.pi * r / 180 for r in [0, 30, 60, 90, 120, 150]]
    _PHASES = [0, np.pi / 2]

    @classmethod
    def general_noise(cls, img_size: int) -> tuple[np.ndarray]:
        patches = np.zeros((img_size, img_size, len(cls._SCALES) * len(cls._ORIENTATIONS) * len(cls._PHASES)), dtype="float32")
        weights = np.zeros((img_size, img_size, len(cls._SCALES) * len(cls._ORIENTATIONS) * len(cls._PHASES)), dtype="float32")

        col = 0
        idx = 0
        uniform_weights = np.random.uniform(-1, 1, size = np.sum(len(cls._ORIENTATIONS) * len(cls._PHASES) * np.array([2 ** i for i in range(len(cls._SCALES))]) ** 2))
        for scale in cls._SCALES:
            for orientation in cls._ORIENTATIONS:
                for phase in cls._PHASES:
                    size = int(img_size / scale)
                    sinusoid = np.repeat([np.linspace(0, 2, size)], size, axis = 0)
                    sinusoid = (sinusoid * np.cos(orientation) + sinusoid.T * np.sin(orientation)) * 2 * np.pi
                    sinusoid = 1 * np.sin(sinusoid + phase)
                    sinusoid = np.tile(sinusoid, (scale, scale))
                    patches[:, :, col] = sinusoid
                    
                    for c in range(scale):
                        for r in range(scale):
                            weights[(size * r):(size * (r + 1) + 1),
                                    (size * c):(size * (c + 1) + 1),
                                    col] = uniform_weights[idx]
                            idx += 1
                    col += 1
        return patches, weights

    @classmethod
    def normalized_noise(cls, noise: np.ndarray) -> np.ndarray:
        noise = noise.copy()
        m = (np.max(noise) + np.min(noise)) / 2
        noise = noise - m
        return noise / np.max(np.abs(noise))

    @classmethod
    def general_noise_stand(cls, img_size: int) -> np.ndarray:
        # 将值归一化，范围为 [-1, 1]
        noise = np.multiply(*cls.general_noise(img_size)).mean(axis = 2)
        return cls.normalized_noise(noise)
    
    @classmethod
    def calc_noise_cluster(cls, img: np.ndarray):
        # 使用之前的方法，进行像素丛聚分析
        p_value = 0.05
        sigma = 4
        # 高斯平滑
        smoothed_image = gaussian_filter(img, sigma = sigma)
        # 转换为Z分数
        mean = np.mean(smoothed_image)
        std = np.std(smoothed_image)
        zscore = (smoothed_image - mean) / std

        threshold = np.percentile(zscore, 100 * (1 - p_value))
        clusters, num_clusters = label(zscore > threshold)
        return zscore, clusters, num_clusters