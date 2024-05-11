import torch
import torchvision
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from datasets.dataset_cfd import CFDDataset

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
    torch.cuda.set_device(2)
    l = (("resnet18", torchvision.models.resnet18, torchvision.models.ResNet18_Weights.DEFAULT),
         ("resnet34", torchvision.models.resnet34, torchvision.models.ResNet34_Weights.DEFAULT),
         ("resnet50", torchvision.models.resnet50, torchvision.models.ResNet50_Weights.DEFAULT),
         ("resnet101", torchvision.models.resnet101, torchvision.models.ResNet101_Weights.DEFAULT),
         ("resnet152", torchvision.models.resnet152, torchvision.models.ResNet152_Weights.DEFAULT))
    
    am = ArchiveModelLayer()

    test_data = CFDDataset(use_train=False, norm=True)
    train_data = CFDDataset(use_train=True, norm=True)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)

    for m_name, m_func, m_param in l:
        model = m_func(weights = m_param).cuda()
        model.layer4.register_forward_hook(am.get_layer4)
        for img, label in tqdm(test_loader):
            model(img.cuda())
        for img, label in tqdm(train_loader):
            model(img.cuda())
        r = torch.concat(am.result, dim = 0).cpu().numpy()
        np.save("work_dirs/study1/{}.npy".format(m_name), r)
        am.reset()

def calc():
    index_x = []
    index_y = []
    for i in range(831):
        for j in range(i + 1):
            if i > j:
                index_x.append(i)
                index_y.append(j)

    data18 = np.load("work_dirs/study1/resnet18.npy")
    data50 = np.load("work_dirs/study1/resnet50.npy")
    data34 = np.load("work_dirs/study1/resnet34.npy")
    data101 = np.load("work_dirs/study1/resnet101.npy")
    data152 = np.load("work_dirs/study1/resnet152.npy")
    d =  np.corrcoef(np.concatenate([np.corrcoef(data18)[index_x,index_y].reshape(1,-1),
                                     np.corrcoef(data34)[index_x,index_y].reshape(1,-1),
                                     np.corrcoef(data50)[index_x,index_y].reshape(1,-1),
                                     np.corrcoef(data101)[index_x,index_y].reshape(1,-1),
                                     np.corrcoef(data152)[index_x,index_y].reshape(1,-1)], axis = 0))
    return pd.DataFrame(d, columns=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"], index=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"])

def stim_calc():
    index_x = []
    index_y = []
    for i in range(831):
        for j in range(i + 1):
            if i > j:
                index_x.append(i)
                index_y.append(j)

    def get_r(*shape: tuple):
        return np.corrcoef(np.random.rand(*shape))

    return np.corrcoef(np.concatenate([get_r(831, 100)[index_x,index_y].reshape(1, -1),
                                       get_r(831, 100)[index_x,index_y].reshape(1, -1)], axis=0))

def render_img_res():
    N = 128
    vals = np.zeros((N * 2, 4))
    vals[:, 0] = np.concatenate([np.linspace(21/255, 255/255, N), np.linspace(255/255, 168/255, N)])
    vals[:, 1] = np.concatenate([np.linspace(43/255, 255/255, N), np.linspace(255/255, 21/255, N)])
    vals[:, 2] = np.concatenate([np.linspace(168/255, 255/255, N), np.linspace(255/255, 43/255, N)])
    vals[:, 3] = 1
    cmap = mpl.colors.ListedColormap(vals)

    fig, axes = plt.subplots(2, 3)
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    for i, mi in enumerate([18, 34, 50, 101, 152]):
        ax: Axes = axes[i//3][i%3]
        d = np.load("work_dirs/study1/resnet{}.npy".format(mi))
        g = sns.clustermap(np.corrcoef(d))
        sns.heatmap(g.data.iloc[g.dendrogram_row.reordered_ind, g.dendrogram_col.reordered_ind],
                    ax = ax, cmap=cmap, cbar = False, vmin=-1, vmax=1)
        ax.set_title("ResNet{}".format(mi), loc="center")
        ax.axis("off")
    ax: Axes = axes[1][2]
    g = sns.clustermap(calc())
    sns.heatmap(g.data.iloc[g.dendrogram_row.reordered_ind, g.dendrogram_col.reordered_ind],
                ax = axes[1][2], cmap=cmap, cbar = True, cbar_ax=cbar_ax, vmin=-1, vmax=1)
    ax.set_yticks([])
    # for mi in [18, 34, 50, 101, 152]:
    #     d = np.load("work_dirs/study1/resnet{}.npy".format(mi))
    #     g = sns.clustermap(np.corrcoef(d), cmap=cmap,
    #                     cbar_pos=(0.9, 0.08, 0.05, 0.71), figsize=(5, 5))
    #     g.ax_heatmap.axes.set_xticks([])
    #     g.ax_heatmap.axes.set_yticks([])
    #     g.ax_col_dendrogram.remove()
    #     g.ax_row_dendrogram.remove()
    #     g.ax_heatmap.set_title("ResNet101", loc = "center")
    #     g.savefig("work_dirs/study1/resnet{}.jpg".format(mi))

    fig.savefig("work_dirs/study1/resnet_compare.jpg", bbox_inches = "tight")

if __name__ == "__main__":
    print(render_img_res())