__all__=['BaseConfig']

class BaseConfig():
    def __init__(self) -> None:
        self.local_rank = 1
        # 单机多卡配置
        self.rank = 0
        self.world_size = 1
        self.seed = 2048

        self.gradient_acc = 1
        self.verbose = 2000
        self.frequent = 10

        # 模型
        self.network = "r50"
        self.embedding_size = 512
        self.resume = False
        self.save_all_states = False
        self.output = "work_dirs"

        # 数据集
        self.rec = "synthetic" # 数据集目录
        self.num_classes = 30 * 10000 # 类别数量
        self.num_image = 100000 # 图像数量

        # dataload numworkers
        self.num_workers = 1

        # 训练参数
        self.num_epoch = 30 # epoch
        self.batch_size = 128
        self.lr = 0.1 # 学习率
        self.fp16 = False

        # 优化器
        self.optimizer = "sgd"
        self.momentum = 0.9
        self.weight_decay = 5e-4

        # CombineMarginLoss
        self.margin_list = (1.0, 0.5, 0.0)

        # Partial FC
        self.sample_rate = 0.1
        self.interclass_filtering_threshold = 0

        # For Large Sacle Dataset, such as WebFace42M
        self.dali = False
        self.dali_aug = False
        
        # calc
        self.total_batch_size = 0
        self.warmup_step = 0
        self.total_step = 0
