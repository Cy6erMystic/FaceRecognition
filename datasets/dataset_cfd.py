import os
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

class CFDDataset(Dataset):
    def __init__(self, root_dir: str = "/media/s5t/caai2024/datasets/1",
                 col = "R011") -> None:
        super(CFDDataset, self).__init__()
        self._root_dir = root_dir
        self._df = pd.read_csv(os.path.join(root_dir, "label_used.csv"))
        self._col = col
        self._col_mean = self._df[col].mean()
        self._col_std = self._df[col].std()

    def __getitem__(self, index):
        row = self._df.iloc[index, :]
        img = cv2.imread(os.path.join(self._root_dir, "face", row["filename"]), cv2.IMREAD_COLOR)
        return np.float32(img).transpose([2, 0, 1]), 1 if row[self._col] > self._col_mean else 0

    def __len__(self):
        return self._df.shape[0]