import os
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CFDDataset(Dataset):
    def __init__(self, root_dir: str = "../../datasets/1",
                 col = "R011", use_train: bool = True) -> None:
        super(CFDDataset, self).__init__()
        self._root_dir = root_dir
        self._df = pd.read_csv(os.path.join(root_dir, "label_used.csv"))
        train, test = train_test_split(self._df.index, train_size = 512,
                                       random_state = 555, shuffle = True)
        self._idx = train if use_train else test
        self._col = col
        self._col_mean = self._df[col].mean()
        self._col_std = self._df[col].std()

    def __getitem__(self, index):
        row = self._df.iloc[self._idx[index], :]
        img = cv2.imread(os.path.join(self._root_dir, "face", row["filename"]), cv2.IMREAD_COLOR)
        # rgb, w, h
        return np.float32(img).transpose([2, 0, 1]), 1 if row[self._col] > self._col_mean else 0

    def __len__(self):
        return len(self._idx)