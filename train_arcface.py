import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models

from model import init_distributed
from model.loss.CombinedMarginLoss import CombinedMarginLoss
from model.loss.PartialFC import PartialFC_V2
from utils import verification
from utils.manager import ModelManager
from datasets.dataset_cfd import CFDDataset

from configs import getLogger
from configs.arcface import ArcFaceConfig

def load_test_datasets(rec: str, col: str):
    test_set = CFDDataset(rec, col=col, use_train = False)
    test_loader = DataLoader(dataset = test_set, batch_size = 32)

    imgs = []
    labels = []
    for img, label in test_loader:
        imgs.append(img)
        labels.append(label)
    imgs = torch.concat(imgs, axis = 0)
    labels = torch.concat(labels, axis = 0)

    imgs_ = []
    labels_: list[torch.Tensor] = []
    for i in range(imgs.shape[0]):
        for j in range(i, imgs.shape[0]):
            if i == j:
                continue
            imgs_.append(imgs[i].unsqueeze(dim = 0))
            imgs_.append(imgs[j].unsqueeze(dim = 0))
            labels_.append(int(labels[i]) == int(labels[j]))
    imgs_ = torch.concat(imgs_, dim = 0)
    return (imgs_, imgs_), labels_

def train(mc: ArcFaceConfig):
    mm = ModelManager()
    logger = getLogger("train_arcface")
    init_distributed(mc.rank, mc.world_size, "tcp://127.0.0.1:12587")
    torch.cuda.set_device(mc.local_rank)
    os.makedirs(mc.output, exist_ok=True)

    train_set = CFDDataset(mc.rec, col=mc.col, random_seed = mc.seed)
    test_set = load_test_datasets(mc.rec, mc.col)

    train_loader = DataLoader(dataset=train_set, batch_size=mc.batch_size, shuffle=True)

    backbone = models.resnet101(weights = models.ResNet101_Weights.DEFAULT).cuda()

    margin_loss = CombinedMarginLoss(64, 1, 0.5, 0)
    module_partial_fc = PartialFC_V2(margin_loss=margin_loss, embedding_size=mc.embedding_size,
                                     num_classes=mc.num_classes, sample_rate=mc.sample_rate, fp16=mc.fp16)
    module_partial_fc.train().cuda()
    optimizer = torch.optim.AdamW(params = [{"params": backbone.parameters()}],
                                  lr = mc.lr, weight_decay = mc.weight_decay)

    for epoch in range(mc.num_epoch):
        mm.set_epoch(epoch + 1)
        mm.reset_loss()
        logger.info("start epoch: {}".format(epoch + 1))
        backbone.train()
        for img, local_labels in tqdm(train_loader):
            local_embeddings = backbone(img.cuda())
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels.cuda())
            mm.update_loss(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % mc.verbose == 0:
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test2(test_set, backbone=backbone,
                                                                                batch_size=mc.batch_size, 
                                                                                nfolds=10)
            mm.update_acc(acc2, std2)
            logger.info(mm)
            if mm.is_best_acc:
                logger.info("SAVING BEST MODEL")
                m1_p = os.path.join(mc.output, "bestAcc_model_backbone.pt")
                torch.save(backbone, m1_p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="FaceRecognitionParser",
                                     description="人脸识别",
                                     epilog="Mupsy@2024")
    parser.add_argument("-c", "--cuda", default=2, type=int)
    args = parser.parse_args()
    mc = ArcFaceConfig({
        "local_rank": args.cuda,
        "col": "R011",
        "output": "work_dirs/form_model_train/arcface/R011",
        "rec": "../../datasets/1",
        "num_epoch": 50000,
        "batch_size": 175,
        "lr": 1e-5,
        "weight_decay": 5e-4
    })
    train(mc)