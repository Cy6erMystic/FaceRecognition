__all__=['BaseConfig', 'BaseModelConfig', 'ModelChoose']

class ModelChoose():
    def __init__(self, model_name, param1, param2, param3, col_name) -> None:
        self.model_name: str = model_name
        self.param1: int = param1
        self.param2: float = param2
        self.param3: float = param3
        self.col_name: str = col_name

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
        self.early_stop = 2000 # 早停
        self.local_rank = 0 # 在哪个GPU训练
        # 训练参数
        self.num_epoch = 50000 # epoch
        self.batch_size = 175
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