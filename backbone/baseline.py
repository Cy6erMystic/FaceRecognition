import torch
import torch.nn as nn
import torchvision.models as models

class FaceRecognitionBase(nn.Module):
    def __init__(self, features: int, num_labels: int) -> None:
        super(FaceRecognitionBase, self).__init__()
        # 1000个特征向量
        self.fc1 = nn.Linear(features, features * num_labels)
        self.fc2 = nn.Linear(features * num_labels, num_labels)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.fc2(x)
        return x