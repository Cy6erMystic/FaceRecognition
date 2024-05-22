import torch
import torch.nn as nn
import torchvision.models as models

class FaceRecognitionBase(torch.nn.Module):
    def __init__(self, features: int, num_labels: int) -> None:
        super(FaceRecognitionBase, self).__init__()
        # 1000个特征向量
        self.fc1 = torch.nn.Linear(features, features * num_labels)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(features * num_labels, num_labels)
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x: torch.Tensor):
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x