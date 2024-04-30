import os
import time
import torch
import argparse
from torch.utils.data import DataLoader

from configs.vae import VAEConfig
from backbone.vae import VAE
from datasets.dataset_cfd import CFDDataset

def main(mc: VAEConfig):
    torch.manual_seed(mc.seed)
    torch.cuda.set_device(mc.local_rank)

    dataset = CFDDataset(mc.rec, mc.col)
    data_loader = DataLoader(dataset = dataset,
                             batch_size = mc.batch_size,
                             shuffle = True)
    
    backbone = VAE(encoder_layer_sizes = mc.encoder_layer_sizes,
                   decoder_layer_sizes = mc.decoder_layer_sizes,
                   latent_size = mc.latent_size,
                   conditional = mc.conditional,
                   num_labels = 0).cuda(mc.local_rank)
    optimizer = torch.optim.Adam(backbone.parameters(), lr = mc.lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", default=2, type=int)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(VAEConfig(args))