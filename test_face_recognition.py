import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models

from datasets.dataset_cfd import CFDDataset
from configs.arcface import ArcFaceConfig

@torch.no_grad()
def save_embed(mc: ArcFaceConfig):
    torch.cuda.set_device(2)
    backbone: torch.nn.Module = torch.load("work_dirs/test/R017/lmcl/1/0.0/0.5/bestAcc_model_backbone.pt")
    backbone.cuda()
    backbone.eval()

    train_set = CFDDataset(mc.rec, col=mc.col, random_seed = mc.seed)
    train_loader = DataLoader(dataset=train_set, batch_size=mc.batch_size, shuffle=True)

    embeddings = []
    labels = []
    for img, local_labels in tqdm(train_loader):
        local_embeddings: torch.Tensor = backbone(img.cuda())
        embeddings.append(local_embeddings.cpu())
        labels.append(local_labels.cpu())
    embeddings = torch.concat(embeddings, dim = 0)
    labels = torch.concat(labels, dim = 0)

    nf = embeddings[labels == 0].mean(dim = 0)
    f = embeddings[labels == 1].mean(dim = 0)

    torch.save(torch.concat([nf.unsqueeze(0), f.unsqueeze(0)], dim = 0), "work_dirs/test/R017/lmcl/1/0.0/0.5/feature.pt")

@torch.no_grad()
def eval_img(img: str):
    torch.cuda.set_device(2)
    backbone: torch.nn.Module = torch.load("work_dirs/test/R017/lmcl/1/0.0/0.5/bestAcc_model_backbone.pt")
    backbone.cuda()
    backbone.eval()

    im = cv2.imread(img, cv2.IMREAD_COLOR)
    im = ((im / 255) - 0.5) / 0.5

    embedings = torch.load("work_dirs/test/R017/lmcl/1/0.0/0.5/feature.pt")
    embeding = backbone(torch.tensor(im.transpose([2, 0, 1]), dtype=torch.float32).cuda().unsqueeze(0)).cpu()
    return torch.sum(torch.square(embedings - embeding), 1)

if __name__ == "__main__":
    # save_embed(ArcFaceConfig({"rec": "../../datasets/1", "col": "R017"}))
    print(eval_img("../../datasets/1/face/0_CFD-WF-014-002-N.jpg"))