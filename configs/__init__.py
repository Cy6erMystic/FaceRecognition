from .base import BaseConfig
from .logging import getLogger_ as getLogger

__all__ = ["getLogger", "ModelConfig"]

class ModelConfig(BaseConfig):
    def __init__(self, args: dict) -> None:
        super().__init__()

        for k, v in args.items():
            if v is not None:
                setattr(self, k, v)

        setattr(self, "total_batch_size", self.batch_size * self.world_size)
        setattr(self, "warmup_step", self.num_image // self.total_batch_size * self.warmup_step)
        setattr(self, "total_step", self.num_image // self.total_batch_size * self.num_epoch) 