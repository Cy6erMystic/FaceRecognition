import os
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from torch.utils.data import DataLoader
from torchvision.models import ResNet

from datasets.dataset_cfd import CFDDataset

def _calc_r(arr: list):
    return np.corrcoef(_calc_raw(arr))

def _calc_raw(arr: list, iter = range(831)):
    index_x = []
    index_y = []
    for i in iter:
        for j in range(i + 1):
            if i > j:
                index_x.append(i)
                index_y.append(j)
    return np.concatenate([np.corrcoef(a)[index_x, index_y].reshape(1, -1) for a in arr], axis=0)

class ArchiveModelLayer():
    def __init__(self) -> None:
        self._result = []
    
    @property
    def result(self):
        return self._result
    
    def get_layer4(self, module, input, output: torch.Tensor):
        self._result.append(output.view(output.shape[0], -1))
    
    def reset(self):
        self._result = []

@torch.no_grad()
def main():
    torch.cuda.set_device(1)
    am = ArchiveModelLayer()

    dataset = CFDDataset(split = False)
    dataloader = DataLoader(dataset = dataset, batch_size = 32, shuffle = False)
    
    model_path = ["./work_dirs/test/R017/softmax/0/0.0/0.0/bestAcc_model_backbone.pt",
                  "./work_dirs/test/R017/cosface/2/0.0/0.0/bestAcc_model_backbone.pt",
                  "./work_dirs/test/R017/lmcl/1/0.0/0.5/bestAcc_model_backbone.pt",
                  "./work_dirs/test/R017/arcface/1/0.8/0.0/bestAcc_model_backbone.pt"]
    for i, model_p in enumerate(model_path):
        model: ResNet = torch.load(model_p).cuda()
        model.eval()
        model.layer4.register_forward_hook(am.get_layer4)
        for img, label in tqdm(dataloader):
            model(img.cuda())
        r = torch.concat(am.result, dim = 0).cpu().numpy()
        np.save("work_dirs/study2/{}.npy".format(i + 1), r)
        am.reset()

def render_img(arr: list):
    N = 128
    vals = np.zeros((N * 2, 4))
    vals[:, 0] = np.concatenate([np.linspace(21/255, 255/255, N), np.linspace(255/255, 168/255, N)])
    vals[:, 1] = np.concatenate([np.linspace(43/255, 255/255, N), np.linspace(255/255, 21/255, N)])
    vals[:, 2] = np.concatenate([np.linspace(168/255, 255/255, N), np.linspace(255/255, 43/255, N)])
    vals[:, 3] = 1
    cmap = mpl.colors.ListedColormap(vals)

    fig, axes = plt.subplots(4, 4, figsize = (14, 10))
    fig.subplots_adjust(right=0.85)
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))

    raw_list = _calc_raw([np.load(x) for x in arr])
    r_list = np.corrcoef(raw_list)
    for i in range(4):
        for j in range(4):
            if i == j:
                # 绘制RSA
                d = np.load(arr[i])
                r = np.corrcoef(d)
                ax: Axes = axes[i][j]
                g = sns.clustermap(r)
                sns.heatmap(g.data.iloc[g.dendrogram_row.reordered_ind, g.dendrogram_col.reordered_ind],
                            ax = ax, cmap=cmap, cbar = False, vmin=-1, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 3:
                    ax.set_xlabel(["Softmax", "SphereFace-2", "CosFace-0.5", "ArcFace-0.8"][j],
                                  labelpad = 0.1)
                if j == 0:
                    ax.set_ylabel(["Softmax", "SphereFace-2", "CosFace-0.5", "ArcFace-0.8"][i])
            else:
                # 绘制散点图
                ax: Axes = axes[i][j]
                ax.scatter(raw_list[i], raw_list[j], s = 0.01, alpha = 0.1, color = sm.to_rgba(r_list[i][j]))
                ax.text(0.75, 0.95, "r={:.2f}".format(r_list[i][j])).set_fontsize('medium')
                if i < 3:
                    ax.set_xticks([])
                if j > 0:
                    ax.set_yticks([])
                if i == 3:
                    ax.set_xlabel(["Softmax", "SphereFace-2", "CosFace-0.5", "ArcFace-0.8"][j])
                if j == 0:
                    ax.set_ylabel(["Softmax", "SphereFace-2", "CosFace-0.5", "ArcFace-0.8"][i])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.01, aspect=30)
    cbar.set_label('Correlation')
    fig.savefig("work_dirs/study2/resnet_compare.jpg", bbox_inches = "tight")

if __name__ == "__main__":
    render_img(["./work_dirs/study2/1.npy",
                "./work_dirs/study2/2.npy",
                "./work_dirs/study2/3.npy",
                "./work_dirs/study2/4.npy"])