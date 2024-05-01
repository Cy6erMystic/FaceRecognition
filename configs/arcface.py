from .base import BaseModelConfig
__all__=['ArcFaceConfig']

class ArcFaceConfig(BaseModelConfig):
    def __init__(self, args: dict = {}) -> None:
        # 单机多卡配置
        self.rank = 0
        self.world_size = 1

        self.gradient_acc = 1
        self.verbose = 2000
        self.frequent = 10

        # 模型
        self.network = "r50"
        self.embedding_size = 1000
        self.resume = False
        self.save_all_states = False
        self.output = "work_dirs"

        # 数据集
        self.rec = "synthetic" # 数据集目录
        self.num_classes = 2 # 类别数量
        self.num_image = 700 # 图像数量

        # dataload numworkers
        self.num_workers = 1

        # 训练参数
        self.fp16 = False

        # CombineMarginLoss
        self.margin_list = (1.0, 0.5, 0.0)

        # Partial FC
        self.sample_rate = 1.0
        self.interclass_filtering_threshold = 0

        # For Large Sacle Dataset, such as WebFace42M
        self.dali = False
        self.dali_aug = False
        
        # calc
        self.total_batch_size = 0
        self.warmup_step = 0
        self.total_step = 0
        super(ArcFaceConfig, self).__init__(args)

        setattr(self, "total_batch_size", self.batch_size * self.world_size)
        setattr(self, "warmup_step", self.num_image // self.total_batch_size * self.warmup_step)
        setattr(self, "total_step", self.num_image // self.total_batch_size * self.num_epoch) 