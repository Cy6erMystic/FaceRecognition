import os
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CFDDataset(Dataset):
    def __init__(self, root_dir: str = "../../datasets/1",
                 col = "R011", use_train: bool = True, norm: bool = True,
                 random_seed: int = 555, split = True) -> None:
        super(CFDDataset, self).__init__()
        self._root_dir = root_dir
        self._norm = norm
        self._df = pd.read_csv(os.path.join(root_dir, "label_used.csv"))

        train, test = train_test_split(self._df.index, train_size = 700,
                                       random_state = random_seed, shuffle = True)
        self._idx = train if use_train else test
        if not split:
            self._idx = self._df.index
        self._preload()

        self._col = col
        self._col_mean = self._df[col].mean()
        self._col_std = self._df[col].std()

    def __getitem__(self, index):
        row = self._df.iloc[self._idx[index], :]
        # img = cv2.imread(os.path.join(self._root_dir, "face", row["filename"]), cv2.IMREAD_COLOR)
        img = self._imgs[index]
        # rgb, w, h
        return np.float32(img).transpose([2, 0, 1]), 1 if row[self._col] > self._col_mean else 0

    def __len__(self):
        return len(self._idx)

    def _preload(self):
        imgs = []
        for i, row in self._df.iloc[self._idx, :].iterrows():
            img_path = os.path.join(self._root_dir, "face", row["filename"])
            img = self._get_img(img_path)
            imgs.append(img)
        self._imgs = imgs

    def _get_img(self, path: str):
        if not self._norm:
            return cv2.imread(path, cv2.IMREAD_COLOR)
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        return ((im / 255) - 0.5) / 0.5