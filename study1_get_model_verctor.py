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

def _random(ndarr: np.ndarray):
    t = np.copy(ndarr)
    for i in range(t.shape[1]):
        np.random.shuffle(t[:, i])
    return t
    # t = np.copy(ndarr).reshape(-1)
    # np.random.shuffle(t)
    # return t.reshape(*ndarr.shape)

def _get_cmap():
    N = 128
    vals = np.zeros((N * 2, 4))
    vals[:, 0] = np.concatenate([np.linspace(21/255, 255/255, N), np.linspace(255/255, 168/255, N)])
    vals[:, 1] = np.concatenate([np.linspace(43/255, 255/255, N), np.linspace(255/255, 21/255, N)])
    vals[:, 2] = np.concatenate([np.linspace(168/255, 255/255, N), np.linspace(255/255, 43/255, N)])
    vals[:, 3] = 1
    cmap = mpl.colors.ListedColormap(vals)
    return cmap

def get_df_long(res, layer_name: str, model_name: str):
    return pd.melt(pd.DataFrame({
        "face_id": [i + 1 for i in range(831)],
        "model_name": [model_name] * 831,
        "layer_name": [layer_name] * 831
    }).merge(pd.DataFrame(res), how = "left", left_index=True, right_index=True), id_vars=["face_id", "model_name", "layer_name"])

@torch.no_grad()
def main():
    torch.cuda.set_device(1)
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
        # model = torch.load("./work_dirs/test/R017/lmcl/1/0.0/0.5/bestAcc_model_backbone.pt").cuda()
        model.eval()
        model.layer4.register_forward_hook(am.get_layer4)
        for img, label in tqdm(test_loader):
            model(img.cuda())
        for img, label in tqdm(train_loader):
            model(img.cuda())
        r = torch.concat(am.result, dim = 0).cpu().numpy()
        np.save("work_dirs/study1/{}.npy".format(m_name), r)
        am.reset()

def calc():
    resnet_arr = [np.load("work_dirs/study1/resnet18.npy"),
                  np.load("work_dirs/study1/resnet34.npy"),
                  np.load("work_dirs/study1/resnet50.npy"),
                  np.load("work_dirs/study1/resnet101.npy"),
                  np.load("work_dirs/study1/resnet152.npy")]
    return pd.DataFrame(_calc_r(resnet_arr), columns=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"], index=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"])

def stim_calc():
    def get_r(*shape: tuple):
        return np.corrcoef(np.random.rand(*shape))
    return _calc_r([get_r(831, 100), 
                    get_r(831, 100), 
                    get_r(831, 100)])

def render_img_res():
    cmap = _get_cmap()

    fig, axes = plt.subplots(2, 3, figsize = (18, 10))
    fig.subplots_adjust(right=0.85)
    for i, mi in enumerate([18, 34, 50, 101, 152]):
        ax: Axes = axes[i//3][i%3]
        d = np.load("work_dirs/study1/resnet{}.npy".format(mi))
        g = sns.clustermap(np.corrcoef(d))
        sns.heatmap(g.data.iloc[g.dendrogram_row.reordered_ind, g.dendrogram_col.reordered_ind],
                    ax = ax, cmap=cmap, cbar = False, vmin=-1, vmax=1)
        ax.set_title("ResNet{}".format(mi), loc="center")
        ax.text(0, 0, "abcde"[i]).set_fontsize('large')
        ax.axis("off")
    ax: Axes = axes[1][2]
    g = sns.clustermap(calc())
    sns.heatmap(g.data.iloc[g.dendrogram_row.reordered_ind, g.dendrogram_col.reordered_ind],
                ax = axes[1][2], cmap=cmap, cbar = False, vmin=-1, vmax=1)
    ax.set_yticks([])
    ax.set_xticklabels(ax.get_xticklabels(), rotation = -30, ha = 'right')
    ax.text(0, 0, "f").set_fontsize('large')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), pad=0.01, aspect=30)
    cbar.set_label('Correlation')
    fig.savefig("work_dirs/study1/resnet_compare.jpg", bbox_inches = "tight")

def permutation_test():
    resnet_arr = [np.load("work_dirs/study1/resnet18.npy"),
                  np.load("work_dirs/study1/resnet34.npy"),
                  np.load("work_dirs/study1/resnet50.npy"),
                  np.load("work_dirs/study1/resnet101.npy"),
                  np.load("work_dirs/study1/resnet152.npy")]

    r = []
    for _ in tqdm(range(10000)):
        r.append(_calc_r([_random(ra) for ra in resnet_arr]))
    r = np.array(r)
    p = np.sum(_calc_r(resnet_arr) < r, axis = 0) / r.shape[0]
    return p

def analysis_method2():
    resnet_arr1 = [np.load("work_dirs/study1_layer1/resnet18.npy"),
                   np.load("work_dirs/study1_layer1/resnet34.npy"),
                   np.load("work_dirs/study1_layer1/resnet50.npy"),
                   np.load("work_dirs/study1_layer1/resnet101.npy"),
                   np.load("work_dirs/study1_layer1/resnet152.npy")]
    resnet_arr2 = [np.load("work_dirs/study1_layer2/resnet18.npy"),
                   np.load("work_dirs/study1_layer2/resnet34.npy"),
                   np.load("work_dirs/study1_layer2/resnet50.npy"),
                   np.load("work_dirs/study1_layer2/resnet101.npy"),
                   np.load("work_dirs/study1_layer2/resnet152.npy")]
    resnet_arr3 = [np.load("work_dirs/study1_layer3/resnet18.npy"),
                   np.load("work_dirs/study1_layer3/resnet34.npy"),
                   np.load("work_dirs/study1_layer3/resnet50.npy"),
                   np.load("work_dirs/study1_layer3/resnet101.npy"),
                   np.load("work_dirs/study1_layer3/resnet152.npy")]
    resnet_arr4 = [np.load("work_dirs/study1_layer4/resnet18.npy"),
                   np.load("work_dirs/study1_layer4/resnet34.npy"),
                   np.load("work_dirs/study1_layer4/resnet50.npy"),
                   np.load("work_dirs/study1_layer4/resnet101.npy"),
                   np.load("work_dirs/study1_layer4/resnet152.npy")]
    calc_arr1 = 1 - _calc_raw(resnet_arr1)
    calc_arr2 = 1 - _calc_raw(resnet_arr2)
    calc_arr3 = 1 - _calc_raw(resnet_arr3)
    calc_arr4 = 1 - _calc_raw(resnet_arr4)

    df = pd.concat([pd.DataFrame({"model_name": ["18", "34", "50", "101", "152"], "layer": [1] * 5}).merge(pd.DataFrame(calc_arr1), how = "right", left_index=True, right_index=True),
                    pd.DataFrame({"model_name": ["18", "34", "50", "101", "152"], "layer": [2] * 5}).merge(pd.DataFrame(calc_arr2), how = "right", left_index=True, right_index=True),
                    pd.DataFrame({"model_name": ["18", "34", "50", "101", "152"], "layer": [3] * 5}).merge(pd.DataFrame(calc_arr3), how = "right", left_index=True, right_index=True),
                    pd.DataFrame({"model_name": ["18", "34", "50", "101", "152"], "layer": [4] * 5}).merge(pd.DataFrame(calc_arr4), how = "right", left_index=True, right_index=True)])
    df = df.reset_index(drop=True)
    df_ = df.iloc[:, 2:]

    cos_r = np.ones(shape = (df_.shape[0], df_.shape[0]))
    for i in range(df_.shape[0]):
        for j in range(i, df_.shape[0]):
            if i == j:
                continue
            # cos = np.sum(np.multiply(df_.iloc[i, 0:], df_.iloc[j, 0:])) / (np.sum(df_.iloc[i, 0:]**2) ** 0.5 * np.sum(df_.iloc[j, 0:]**2) ** 0.5)
            cos = np.corrcoef(df_.iloc[i, :], df_.iloc[j, :])[0][1]
            cos_r[i][j] = cos
            cos_r[j][i] = cos

    # g = sns.clustermap(cos_r)
    df_i = df
    s = []
    for i, row in df_i.iterrows():
        s.append(f"ResNet{row['model_name']}_Layer{row['layer']}")

    cmap = _get_cmap()
    fig = plt.figure(figsize=(5,5))
    ax = sns.heatmap(cos_r, cmap = cmap, cbar = False, vmin = 0, vmax = 1)
    ax.set_xticks(ticks = [i + 0.5 for i in range(len(s))],
                labels = s, rotation = 90)
    ax.set_yticks(ticks = [i + 0.5 for i in range(len(s))],
                labels = s, rotation = 0)

    cax = fig.add_axes([-0.12, -0.03, 0.2, 0.05])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cb = fig.colorbar(sm, cax = cax, orientation='horizontal')
    fig.savefig("work_dirs/study1/resnet_compare_v2.jpg", bbox_inches = "tight")

if __name__ == "__main__":
    # main()
    analysis_method2()