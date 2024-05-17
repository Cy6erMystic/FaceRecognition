from __future__ import annotations
import numpy as np

class RCIC():
    _SCALES = [2 ** i for i in range(5)]
    _ORIENTATIONS = [np.pi * r / 180 for r in [0, 30, 60, 90, 120, 150]]
    _PHASES = [0, np.pi / 2]
    def __init__(self) -> None:
        pass

    @classmethod
    def general_noise(cls, img_size: int):
        patches = np.zeros((img_size, img_size, len(cls._SCALES) * len(cls._ORIENTATIONS) * len(cls._PHASES)), dtype="float32")
        weights = np.zeros((img_size, img_size, len(cls._SCALES) * len(cls._ORIENTATIONS) * len(cls._PHASES)), dtype="float32")

        col = 0
        idx = 0
        d = np.random.uniform(-1, 1, size = np.sum(len(cls._ORIENTATIONS) * len(cls._PHASES) * np.array([2 ** i for i in range(len(cls._SCALES))]) ** 2))
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
                                    col] = d[idx]
                            idx += 1
                    col += 1
        return patches, weights