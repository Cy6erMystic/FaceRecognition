import torch
import random
import numpy as np
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.labels = [0,1,2,3,4]

    def __getitem__(self, index):
        return self.img, random.choice(self.labels)

    def __len__(self):
        return 1000000