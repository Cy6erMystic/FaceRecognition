import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from datasets.dataset_cfd import CFDDataset
from utils.rcic import RCIC

def step1():
    # 获取差异最小的图片, 也就是模型识别效果不好的
    torch.cuda.set_device(2)
    train_set = CFDDataset(col="R017", split=False)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=False)

    backbone: torch.nn.Module = torch.load("work_dirs/test/R017/lmcl/1/0.0/0.5/bestAcc_model_backbone.pt")
    backbone.cuda()
    backbone.eval()

    embeddings = []
    labels = []
    with torch.no_grad():
        for img, local_labels in tqdm(train_loader):
            local_embeddings: torch.Tensor = backbone(img.cuda())
            embeddings.append(local_embeddings.cpu())
            labels.append(local_labels.cpu())
    embeddings: torch.Tensor = torch.concat(embeddings, axis = 0)
    labels: torch.Tensor = torch.concat(labels, axis = 0)

    feature_class = torch.concat([embeddings[labels == 0].mean(dim = 0).unsqueeze(0),
                                  embeddings[labels == 1].mean(dim = 0).unsqueeze(0)], axis = 0)

    coss = []
    for i in range(embeddings.shape[0]):
        cos = torch.cosine_similarity(embeddings[i], feature_class, axis = 1).unsqueeze(0)
        coss.append(cos)
    coss = torch.concat(coss, dim = 0)
    # 得到模型识别效果不好的id索引 以及文件名称
    return coss.max(dim = 1).values.argmin(), \
           train_set._df["filename"][int(coss.max(dim = 1).values.argmin())]

def step2():
    filename = "0_CFD-LF-236-221-N.jpg"
    im = cv2.imread("../../datasets/1/face/{}".format(filename), cv2.IMREAD_COLOR) # bgr
    im: np.ndarray = ((im / 255) - 0.5) / 0.5

    nosie_patchs, nosie_wights = RCIC.general_noise(112)
    noise = (nosie_patchs * nosie_wights).mean(axis=2) / 0.3

    img = np.zeros(im.shape)
    img[:,:,0] = np.mean([im[:,:,0], noise], axis = 0)
    img[:,:,1] = np.mean([im[:,:,1], noise], axis = 0)
    img[:,:,2] = np.mean([im[:,:,2], noise], axis = 0)

if __name__ == "__main__":
    step1()