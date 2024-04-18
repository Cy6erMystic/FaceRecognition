import torch
import math

class CombinedMarginLoss(torch.nn.Module):
    """
    若m1 = 1, 则不进行A-Softmax
    
    若m2 = 0, 则不进行Arc
    
    若m3 = 0, 则不进行LMCL
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