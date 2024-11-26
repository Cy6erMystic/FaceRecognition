import torch
import math
from typing import Callable
from torch.nn.functional import linear, normalize

class CombinedMarginLoss(torch.nn.Module):
    """
    若m1 = 1, 则不进行A-Softmax - SphereFace
    
    若m2 = 0, 则不进行Arc - ArcFace
    
    若m3 = 0, 则不进行LMCL - CosFace
    """
    def __init__(self, 
                 s, 
                 m1: int, # A-Softmax Loss
                 m2, # Arc Margin Cosine Loss
                 m3, # Large Margin Cosine Loss
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # 获取 正类 样本的索引值
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                # 挑出超出类间间距的样本
                dirty = logits > self.interclass_filtering_threshold
                # 将False和True转换为 0 和 1
                dirty = dirty.float()

                # 挑出目标样本和特征，做掩码处理
                # 获取大小为 样本*类别 的矩阵
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                # 将每个目标样本进行掩码遮蔽，也就是替换为0
                mask.scatter_(1, labels[index_positive].view(-1, 1), 0)

                # 也就是，超出阈限，且不为目标样本，则用0替代。意味着类间特征太大了
                dirty[index_positive] *= mask
                # 取反
                tensor_mul = 1 - dirty  
            # 重新分配特征  
            logits = tensor_mul * logits

        # 取出 正类 样本里，对应的目标特征值
        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        with torch.no_grad():
            # 目标特征值取反余弦值
            target_logit.arccos_()
            # 全部特征值取反余弦值
            logits.arccos_()

            # 目标特征值乘以m1
            # 再加上m2
            final_target_logit = self.m1 * target_logit + self.m2
            # 用添加了弧度的目标特征值，替换原特征值
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            # 重新计算余弦值
            logits.cos_()
            # 目标特征值减去m3
            logits[index_positive, labels[index_positive].view(-1)] -= self.m3
        # 对于余弦值进行缩放
        logits = logits * self.s

        return logits


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """ """
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # local to global
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        logits.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)


class CalcLoss(torch.nn.Module):
    def __init__(self, margin_loss: Callable,
                 embedding_size: int,
                 num_classes: int) -> None:
        super(CalcLoss, self).__init__()
        self.margin_softmax = margin_loss
        self.dist_cross_entropy = DistCrossEntropy()
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))

    def forward(self, local_embeddings: torch.Tensor, local_labels: torch.Tensor):
        norm_embeddings = normalize(local_embeddings)
        norm_weight_activated = normalize(self.weight)
        logits = linear(norm_embeddings, norm_weight_activated)

        logits = logits.clamp(-1, 1)
        labels = local_labels.view(-1, 1)
        
        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return loss

if __name__ == "__main__":
    num_samples = 10 # 10个特征
    num_classes = 5 # 5个类别

    # 初始化损失函数
    s = 30.0  # 缩放参数
    m1 = 1  # 第一个边界参数
    m2 = 0.0  # 第二个边界参数，用于ArcFace
    m3 = 0.0  # 第三个边界参数
    interclass_filtering_threshold = 0.5  # 类间过滤阈值

    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))
    loss_fn = CombinedMarginLoss(s, m1, m2, m3, interclass_filtering_threshold)
    print(loss_fn(logits, labels))