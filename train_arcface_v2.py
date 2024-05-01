"""
第二个版本, 主打一个单机多卡, 高效率
- 模型并行, 一块卡一个模型
- 数据并行, 一块卡部分数据
"""
import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from multiprocessing import Process

from datasets.dataset_mx import MXFaceDataset
from utils.distributed_sampler import DistributedSampler, setup_seed, worker_init_fn

from model import init_distributed
from backbone import get_model
from model.loss.CombinedMarginLoss import CombinedMarginLoss
from model.loss.PartialFC import PartialFC_V2
from model.lr.PolynomialLRWarmup import PolynomialLRWarmup

from configs.arcface import ArcFaceConfig
from configs import getLogger

def train(mc: ArcFaceConfig):
    logger = getLogger("train")
    init_distributed(mc.rank, mc.world_size)
    setup_seed(mc.seed, cuda_deterministic=False)
    torch.cuda.set_device(mc.local_rank)
    os.makedirs(mc.output, exist_ok=True)

    train_set = MXFaceDataset(mc.rec)
    train_sampler = DistributedSampler(dataset = train_set,
                                       num_replicas = mc.world_size,
                                       rank = mc.rank,
                                       seed = mc.seed,
                                       shuffle = True)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=mc.batch_size,
                              sampler=train_sampler,
                              num_workers=mc.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              worker_init_fn=lambda x: worker_init_fn(worker_id=x,
                                                                      num_workers=mc.num_workers,
                                                                      rank=mc.rank,
                                                                      seed=mc.seed))

    backbone = get_model(mc.network, dropout = 0.0, fp16 = mc.fp16, num_features = mc.embedding_size).cuda(mc.local_rank)
    backbone = torch.nn.parallel.DistributedDataParallel(module=backbone, broadcast_buffers=False,
                                                         bucket_cap_mb=16, device_ids=[mc.local_rank],
                                                         find_unused_parameters=True)
    backbone.train()
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(64, mc.margin_list[0], mc.margin_list[1], mc.margin_list[2], mc.interclass_filtering_threshold)
    module_partial_fc = PartialFC_V2(margin_loss=margin_loss, embedding_size=mc.embedding_size,
                                     num_classes=mc.num_classes, sample_rate=mc.sample_rate, fp16=mc.fp16)
    module_partial_fc.train().cuda(mc.local_rank)
    
    opt = torch.optim.AdamW(params=[{"params": backbone.parameters()}],
                            lr = mc.lr, weight_decay=mc.weight_decay)

    lr_scheduler = PolynomialLRWarmup(optimizer=opt, warmup_iters=mc.warmup_step,
                                      total_iters=mc.total_step)

    start_epoch = 0
    global_step = 0
    if mc.resume:
        dict_checkpoint = torch.load(os.path.join(mc.output, f"checkpoint_gpu_{mc.rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    
    for epoch in range(start_epoch, mc.num_epoch):
        train_sampler.set_epoch(epoch)
        logger.info("当前训练批次: {}".format(epoch))
        for img, local_labels in tqdm(train_loader):
            global_step += 1
            local_embeddings = backbone(img.cuda(mc.local_rank))
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels.cuda(mc.local_rank))

            if mc.fp16:
                amp.scale(loss).backward()
                if global_step % mc.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % mc.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

        if mc.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(mc.output, f"checkpoint_gpu_{mc.rank}.pt"))

        if mc.rank == 0:
            path_module = os.path.join(mc.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="FaceRecognitionParser",
                                     description="人脸识别",
                                     epilog="Mupsy@2024")
    parser.add_argument("-r", "--rank", default=0, type=int)
    parser.add_argument("-p", "--processes", default=1, type=int)
    parser.add_argument("-c", "--cuda", default=2, type=int)
    args = parser.parse_args()
    mc = ArcFaceConfig({
        "rank": args.rank,
        "world_size": args.processes, 
        "local_rank": args.cuda,
        "output": "work_dirs/complex_train",
        "network": "r50",
        "rec": "../../datasets/faces_ms1m-retinaface-t1",
        "num_classes": 93431,
        "num_image": 5179510,
        "num_epoch": 20,
        "num_workers": 1,
        "margin_list": (1, 0.5, 0.0),
        "embedding_size": 512,
        "batch_size": 256,
        "fp16": False,
        "sample_rate": 1.0,
        "save_all_states": True
    })
    train(mc)
