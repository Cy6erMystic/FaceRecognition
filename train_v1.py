"""
第一个版本, 主打一个精简, 啥复杂的东西都不要
主打一个大道至简, 可能存在超显存的问题
"""
import os
import torch
from torch import distributed
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import ModelConfig as mc
from configs import getLogger
from backbone import get_model

from model.loss.CombinedMarginLoss import CombinedMarginLoss
from model.loss.PartialFC import PartialFC_V2
from model import init_distributed

from datasets.dataset_mx import MXFaceDataset

logger = getLogger("train")
def main(args: dict):
    mc.update(args)
    init_distributed(mc.rank, mc.world_size)

    train_set = MXFaceDataset(mc.rec)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=mc.batch_size,
                              shuffle=True)

    backbone = get_model(mc.network, dropout = 0.0, fp16 = mc.fp16, num_features = mc.embedding_size).cuda()
    backbone.train()

    margin_loss = CombinedMarginLoss(64, mc.margin_list[0], mc.margin_list[1], mc.margin_list[2], mc.interclass_filtering_threshold)
    module_partial_fc = PartialFC_V2(margin_loss=margin_loss, embedding_size=mc.embedding_size,
                                     num_classes=mc.num_classes, sample_rate=mc.sample_rate,fp16=mc.fp16)
    module_partial_fc.train().cuda()

    opt = torch.optim.AdamW(params=[{"params": backbone.parameters()}],
                            lr = mc.lr, weight_decay=mc.weight_decay)

    global_step = 0
    for epoch in range(mc.num_epoch):
        logger.info("start train...")
        for img, local_labels in tqdm(train_loader):
            global_step += 1
            local_embeddings = backbone(img.cuda())
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels.cuda())

            loss.backward()
            opt.step()
            opt.zero_grad()
    
        if mc.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict()
            }
            torch.save(checkpoint, os.path.join(mc.output, f"checkpoint_gpu_{mc.rank}.pt"))

        if mc.rank == 0:
            torch.save(backbone.state_dict(), "model.pt")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    main({
        "network": "r50",
        "rec": "/media/s5t/caai2024/datasets/ms1m-retinaface-t1",
        "num_classes": 93431,
        "num_image": 5179510,
        "num_epoch": 20,
        "margin_list": (1, 0.5, 0.0),
        "embedding_size": 512,
        "batch_size": 128,
        "output": "word_dirs/simple_train",
        "fp16": False,
        "sample_rate": 1.0
    })