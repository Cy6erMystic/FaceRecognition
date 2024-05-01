__all__=['BaseConfig']

class BaseConfig():
    def __init__(self, args: dict = {}) -> None:
        self._replace(args)
    
    def _replace(self, args: dict):
        for k, v in args.items():
            if v is not None:
                setattr(self, k, v)

class BaseModelConfig(BaseConfig):
    def __init__(self, args: dict = {}) -> None:
        # 初始化参数
        self.seed = 2048 # 随机数种子
        self.local_rank = 0 # 在哪个GPU训练
        # 训练参数
        self.num_epoch = 30 # epoch
        self.batch_size = 128
        self.lr = 0.1 # 学习率
        # 优化器参数
        self.optimizer = "sgd"
        self.weight_decay = 5e-4
        self.momentum = 0.9
        # 额外参数
        self.verbose = 20 # 间隔多久测试一下
        # 数据集参数
        self.rec = "synthetic"
        self.output = "work_dirs"
        self.col = None
        super().__init__(args)