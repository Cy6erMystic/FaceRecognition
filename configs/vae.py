from .base import BaseConfig

__all__=['VAEConfig']
class VAEConfig(BaseConfig):
    def __init__(self, args: dict = {}) -> None:
        self._init()
        super(VAEConfig, self).__init__(args)
        
    def _init(self):
        self.rec = "../../datasets/1"
        self.col = "R011"

        self.local_rank = 0
        self.seed = 512
        self.epochs = 50000
        self.batch_size = 64
        self.lr = 1e-4
        self.print_every = 100

        self.encoder_layer_sizes = [784, 256]
        self.decoder_layer_sizes = [256, 784]
        self.latent_size = 2
        self.conditional = False