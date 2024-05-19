import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
    embeddings: torch.Tensor = torch.concat(embeddings, dim = 0)
    labels: torch.Tensor = torch.concat(labels, dim = 0)

    feature_class = torch.concat([embeddings[labels == 0].mean(dim = 0).unsqueeze(0),
                                  embeddings[labels == 1].mean(dim = 0).unsqueeze(0)], dim = 0)
    torch.save(feature_class, "work_dirs/test/R017/lmcl/1/0.0/0.5/feature2.pt")

    coss = []
    for i in range(embeddings.shape[0]):
        cos = torch.cosine_similarity(embeddings[i], feature_class, dim = 1).unsqueeze(0)
        coss.append(cos)
    coss = torch.concat(coss, dim = 0)
    # 得到模型识别效果不好的id索引 以及文件名称
    return coss.max(dim = 1).values.argmin(), \
           train_set._df["filename"][train_set._idx[int(coss.max(dim = 1).values.argmin())]]

@torch.no_grad()
def step2():
    # 获取符合要求的随机噪音
    torch.cuda.set_device(2)
    backbone: torch.nn.Module = torch.load("work_dirs/test/R017/lmcl/1/0.0/0.5/bestAcc_model_backbone.pt")
    backbone.cuda()
    backbone.eval()
    features = torch.load("work_dirs/test/R017/lmcl/1/0.0/0.5/feature2.pt")

    get_embed = lambda img_: backbone(torch.tensor(img_.transpose([2,0,1]), dtype=torch.float32).unsqueeze(0).cuda()).cpu()
    get_cosine = lambda embed_: torch.cosine_similarity(embed_, features, dim = 1)

    filename = "0_CFD-LF-236-221-N.jpg"
    im = cv2.imread("../../datasets/1/face/{}".format(filename), cv2.IMREAD_COLOR) # bgr
    im: np.ndarray = ((im / 255) - 0.5) / 0.5

    raw_cos_sim = get_cosine(get_embed(im))
    more = [] # 更像
    less = []
    for _ in tqdm(range(10000)):
        noise = RCIC.general_noise_stand(112)

        img_add = np.zeros(im.shape)
        img_reduce = np.zeros(im.shape)
        img_add[:,:,0] = np.mean([im[:,:,0], noise], axis = 0)
        img_add[:,:,1] = np.mean([im[:,:,1], noise], axis = 0)
        img_add[:,:,2] = np.mean([im[:,:,2], noise], axis = 0)
        img_reduce[:,:,0] = np.mean([im[:,:,0], -noise], axis = 0)
        img_reduce[:,:,1] = np.mean([im[:,:,1], -noise], axis = 0)
        img_reduce[:,:,2] = np.mean([im[:,:,2], -noise], axis = 0)

        noise_cos_sim_add = get_cosine(get_embed(img_add))
        noise_cos_sim_reduce = get_cosine(get_embed(img_reduce))
        if noise_cos_sim_add[1] > noise_cos_sim_reduce[1]:
            more.append(noise)
        else:
            less.append(noise)
    np.save("./work_dirs/study4/more.npy", np.array(more))
    np.save("./work_dirs/study4/less.npy", np.array(less))

def step3():
    filename = "0_CFD-LF-236-221-N.jpg"
    im = cv2.imread("../../datasets/1/face/{}".format(filename), cv2.IMREAD_COLOR) # bgr
    im: np.ndarray = ((im / 255) - 0.5) / 0.5

    noise = RCIC.normalized_noise(np.load("./work_dirs/study4/less.npy").mean(axis = 0))
    img_add = np.zeros(im.shape)
    img_reduce = np.zeros(im.shape)
    img_add[:,:,0] = np.mean([im[:,:,0], noise], axis = 0)
    img_add[:,:,1] = np.mean([im[:,:,1], noise], axis = 0)
    img_add[:,:,2] = np.mean([im[:,:,2], noise], axis = 0)
    img_reduce[:,:,0] = np.mean([im[:,:,0], -noise], axis = 0)
    img_reduce[:,:,1] = np.mean([im[:,:,1], -noise], axis = 0)
    img_reduce[:,:,2] = np.mean([im[:,:,2], -noise], axis = 0)

    plt.figure()
    plt.imshow(img_add[:,:,[2,1,0]] / 2 + 0.5)
    plt.axis("off")
    plt.savefig("./work_dirs/study4/add.jpg", bbox_inches = "tight")

    plt.figure()
    plt.imshow(img_reduce[:,:,[2,1,0]] / 2 + 0.5)
    plt.axis("off")
    plt.savefig("./work_dirs/study4/reduce.jpg", bbox_inches = "tight")

def step4():
    filename = "0_CFD-LF-236-221-N.jpg"
    im = cv2.imread("../../datasets/1/face/{}".format(filename), cv2.IMREAD_COLOR) # bgr
    im: np.ndarray = ((im / 255) - 0.5) / 0.5

    cmap1 = ListedColormap([[0,0,0,1],
                           [1,0,0,1]])
    cmap2 = ListedColormap([[0,0,0,1],
                           [0,1,0,1]])

    z1,c1,n1 = RCIC.calc_noise_cluster(np.load("./work_dirs/study4/more.npy").mean(axis = 0))
    z2,c2,n2 = RCIC.calc_noise_cluster(np.load("./work_dirs/study4/less.npy").mean(axis = 0))

    plt.figure()
    plt.axis("off")
    plt.imshow((im[:,:,[2,1,0]] + 1) / 2)
    plt.imshow(c1, cmap = cmap1, alpha=0.2)
    plt.imshow(c2, cmap = cmap2, alpha=0.2)
    plt.savefig("./work_dirs/study4/cluster.jpg", bbox_inches = "tight")

if __name__ == "__main__":
    step4()