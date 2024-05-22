import os
import argparse
import torch
import logging
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.models as models
from multiprocessing import Manager, Queue, Pool

from model import init_distributed
from model.loss.PartialFC import PartialFC_V2
from model.loss.CombinedMarginLoss import CombinedMarginLoss, CalcLoss
from utils import verification
from utils.manager import ModelManager
from utils.logger import Logger
from backbone.baseline import FaceRecognitionBase
from datasets.dataset_cfd import CFDDataset

from configs.arcface import ArcFaceConfig
from configs.base import ModelChoose

def load_test_datasets(rec: str, col: str, random_seed: int):
    test_set = CFDDataset(rec, col=col, random_seed=random_seed, use_train = False)
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

def train(mc: ArcFaceConfig, mmc: ModelChoose):
    mm = ModelManager()
    logger = Logger(name = "train", mmc = mmc)
    # init_distributed(mc.rank, mc.world_size, "tcp://127.0.0.1:1258{}".format(mc.local_rank))
    torch.cuda.set_device(mc.local_rank)
    os.makedirs(mc.output, exist_ok=True)

    train_set = CFDDataset(mc.rec, col=mc.col, random_seed = mc.seed)
    test_set = load_test_datasets(mc.rec, mc.col, mc.seed)
    train_loader = DataLoader(dataset=train_set, batch_size=mc.batch_size, shuffle=True)

    backbone = models.resnet101(weights = models.ResNet101_Weights.DEFAULT).cuda()

    if mmc.model_name == "softmax":
        predict_model = FaceRecognitionBase(1000, 2).cuda()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params = [{"params": backbone.parameters()},
                                                {"params": predict_model.parameters()}],
                                      lr = mc.lr, weight_decay = mc.weight_decay)
    else:
        margin_loss = CombinedMarginLoss(64, mmc.param1, mmc.param2, mmc.param3)
        # module_partial_fc = PartialFC_V2(margin_loss=margin_loss, embedding_size=mc.embedding_size,
        #                                 num_classes=mc.num_classes, sample_rate=mc.sample_rate, fp16=mc.fp16)
        module_partial_fc = CalcLoss(margin_loss=margin_loss, num_classes=mc.num_classes, embedding_size=mc.embedding_size)
        module_partial_fc.train().cuda()

        optimizer = torch.optim.AdamW(params = [{"params": backbone.parameters()},
                                                {"params": module_partial_fc.parameters()}],
                                      lr = mc.lr, weight_decay = mc.weight_decay)

    duration_best = 0
    for epoch in range(mc.num_epoch):
        mm.set_epoch(epoch + 1)
        mm.reset_loss()
        logger.info("start epoch: {}".format(epoch + 1))
        if mmc.model_name == "softmax":
            predict_model.train()
        backbone.train()
        for img, local_labels in tqdm(train_loader):
            local_embeddings = backbone(img.cuda())
            if mmc.model_name == "softmax":
                loss: torch.Tensor = criterion(predict_model(local_embeddings),
                                               local_labels.cuda())
            else:
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
                duration_best = 0
                logger.info("SAVING BEST MODEL")
                m1_p = os.path.join(mc.output, "bestAcc_model_backbone.pt")
                torch.save(backbone, m1_p)
                if mmc.model_name == "softmax":
                    m2_p = os.path.join(mc.output, "bestAcc_model_predict.pt")
                    torch.save(predict_model, m2_p)
                else:
                    m2_p = os.path.join(mc.output, "bestAcc_model_partial_fc.pt")
                    torch.save(module_partial_fc, m2_p)

        duration_best += 1
        if mc.early_stop < duration_best:
            logger.info("BREAK, the model is under the best.")
            break
    return 0

def run(cuda: int, mmc: ModelChoose):
    mc = ArcFaceConfig({
        "local_rank": cuda,
        "col": mmc.col_name,
        "early_stop": 2000,
        "verbose": 50,
        "output": "work_dirs/test/{}/{}/{}/{}/{}".format(mmc.col_name,
                                                         mmc.model_name,
                                                         mmc.param1,
                                                         mmc.param2,
                                                         mmc.param3),
        "rec": "../../datasets/1",
        "num_epoch": 100000,
        "batch_size": 175,
        "lr": 1e-5,
        "weight_decay": 5e-4,
        "num_classes": 2,
        "embedding_size": 1000
    })
    train(mc, mmc)

if __name__ == "__main__":
    df_v = pd.read_csv("train_a_predict_val.csv")
    df = pd.read_csv("train_a_model_compare.csv")
    parser = argparse.ArgumentParser(prog="FaceRecognitionParser",
                                     description="人脸识别",
                                     epilog="Mupsy@2024")
    parser.add_argument("-c", "--cuda", default=0, type=int)
    args = parser.parse_args()
    cuda: int = args.cuda % torch.cuda.device_count()

    # 全部的softmax重新跑一下，因为出问题了
    for i, row1 in df_v.iloc[cuda * 5:(cuda + 1) * 5].iterrows():
        for j, row2 in df.iloc[[0]].iterrows():
            mmc = ModelChoose(row2["name"], int(row2["param1"]), float(row2["param2"]), 
                              float(row2["param3"]), row1["name"])
            run(cuda, mmc)