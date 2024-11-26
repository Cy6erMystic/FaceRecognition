import os
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from configs.i8n import col_to_name_cn, col_to_name_en
import statsmodels.api as sm

from datasets.dataset_cfd import CFDDataset
from utils.rcic import RCIC
plt.rcParams["font.family"] = 'SimHei'

class study3():
    def __init__(self, col: str, path: str):
        self.col = col

        self.model_path = f"work_dirs/test/{self.col}/{path}/bestAcc_model_backbone.pt"
        self.model_save_feature = f"work_dirs/test/{self.col}/{path}/feature2.pt"
        self.work_path = f"work_dirs/study3/{self.col}/{path}"
        if not os.path.exists(self.work_path):
            os.makedirs(self.work_path)
        print(self.work_path)

    def start(self):
        argmax, img_file = self.step1()
        self.step2(str(img_file))
        self.step3(str(img_file))
        self.step4(str(img_file))

    def merge_img(self):
        fig1, ax1 = plt.subplots(3, 5)
        for i1, col in enumerate(os.listdir("./work_dirs/study3")):
            if os.path.isfile(f"./work_dirs/study3/{col}"):
                continue

            model_name = os.listdir(f"./work_dirs/study3/{col}")[0]
            p1 = os.listdir(f"./work_dirs/study3/{col}/{model_name}")[0]
            p2 = os.listdir(f"./work_dirs/study3/{col}/{model_name}/{p1}")[0]
            p3 = os.listdir(f"./work_dirs/study3/{col}/{model_name}/{p1}/{p2}")[0]
            work_dir = f"./work_dirs/study3/{col}/{model_name}/{p1}/{p2}/{p3}"

            ri = i1 // 5
            ci = i1 % 5
            ax: Axes = ax1[ri][ci]
            im = cv2.imread(f"{work_dir}/cluster.jpg", cv2.IMREAD_COLOR) # bgr
            ax.imshow(im[:, :, [2,1,0]])
            ax.axis("off")
            ax.set_title(col_to_name_en[col])
        fig1.savefig("./work_dirs/study3/cluster.jpg", bbox_inches = "tight")

    def prepare_data_for_aov(self):
        res = []
        f = []
        for i1, col in enumerate(os.listdir("./work_dirs/study3")):
            if os.path.isfile(f"./work_dirs/study3/{col}"):
                continue

            model_name = os.listdir(f"./work_dirs/study3/{col}")[0]
            p1 = os.listdir(f"./work_dirs/study3/{col}/{model_name}")[0]
            p2 = os.listdir(f"./work_dirs/study3/{col}/{model_name}/{p1}")[0]
            p3 = os.listdir(f"./work_dirs/study3/{col}/{model_name}/{p1}/{p2}")[0]
            work_dir = f"./work_dirs/study3/{col}/{model_name}/{p1}/{p2}/{p3}"

            f.append(col)
            res.append(np.load(f"{work_dir}/more.npy").mean(axis = 0))
        res = np.array(res)

        cc = []
        for i1 in range(112):
            for i2 in range(112):
                for j in range(15):
                    cc.append(((i1 * 112 + i2), j, res[j, i1, i2]))

        df = pd.DataFrame(np.array(cc), columns=["pixel", "eval_type", "value"])
        df["pixel"] = df["pixel"].astype("i8")
        df["eval_type"] = df["eval_type"].astype("i8").map(lambda x: list(map(lambda x: col_to_name_cn[x], f))[x])
        df.to_csv("a.csv", index = False)

    def aov_result(self):
        df = pd.read_csv("a.csv")
        # aov分析
        model = sm.formula.ols("value ~ C(eval_type)", data = df).fit()
        anova_table = sm.stats.anova_lm(model, type = 1)

        # 事后比较
        tukey = sm.stats.multicomp.pairwise_tukeyhsd(endog=df["value"], groups=df["eval_type"], alpha=0.05)
        
        matrix = np.zeros((2, 15, 15)) + np.diag(np.repeat([1], 15))
        for i1 in range(tukey.groupsunique.size):
            for i2 in range(i1 + 1, tukey.groupsunique.size):
                pval = tukey.pvalues[len(res)]
                cval = study3.cohen_d(
                    df.query(f"eval_type == '{tukey.groupsunique[i1]}'")["value"].to_numpy(),
                    df.query(f"eval_type == '{tukey.groupsunique[i2]}'")["value"].to_numpy()
                )
                matrix[0][i1][i2] = pval
                matrix[0][i2][i1] = pval
                matrix[1][i1][i2] = np.abs(cval)
                matrix[1][i2][i1] = np.abs(cval)
        # 绘制热力图
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks(ticks=[i for i in range(15)], labels=tukey.groupsunique, rotation = 90)
        ax.set_yticks(ticks=[i for i in range(15)], labels=tukey.groupsunique, rotation = 0)
        ax.set_xlim(xmin=-1, xmax=15)
        ax.set_ylim(ymin=-1, ymax=15)

        colors = np.zeros((15, 15, 4))
        for i in range(matrix[0].shape[0]):
            c_i = 14 - i
            for j in range(matrix[0].shape[1]):
                colors[c_i][j] = [1, 1, 1, 1]
                if matrix[0][i][j] < 0.05:
                    colors[c_i][j] = [0.5, 0, 0, 0.25]
                if matrix[0][i][j] < 0.01:
                    colors[c_i][j] = [1, 0, 0, 0.25]

                size = 0.02 if matrix[1][i][j] < 0.2 else (0.1 if matrix[1][i][j] < 0.8 else 0.2)
                if i != j:
                    cir = plt.Circle((j, c_i), size * 2, color = "#000000", alpha = 0.1)
                    ax.add_patch(cir)
                    ax.text(j, c_i, f"{matrix[1][i][j]:.2f}", fontsize = 8, color = "#000000", ha = 'center', va = "center")
        ax.imshow(colors)
        # legend 1
        ax = fig.add_axes([0.7, 0.72, 0.19, 0.15])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axes.set_axis_off()
        ax.add_patch(plt.Rectangle((0.00, 0.01), 0.99, 0.99, edgecolor = [0, 0, 0, 1], facecolor = [1, 1, 1, 1], fill = True, linewidth = 1))
        ax.add_patch(plt.Rectangle((0.10, 0.80), 0.38, 0.16, edgecolor = [0, 0, 0, 1], facecolor = [1, 1, 1, 1], fill = True, linewidth = 1))
        ax.add_patch(plt.Rectangle((0.10, 0.50), 0.38, 0.16, edgecolor = [0, 0, 0, 1], facecolor = [0.5, 0, 0, 0.25], fill = True, linewidth = 1))
        ax.add_patch(plt.Rectangle((0.10, 0.20), 0.38, 0.16, edgecolor = [0, 0, 0, 1], facecolor = [1, 0, 0, 0.25], fill = True, linewidth = 1))
        ax.text(0.55, 0.88, "p ≥ 0.05", fontsize=10, color="#000000", va="center", ha="left")
        ax.text(0.55, 0.58, "p ＜ 0.05", fontsize=10, color="#000000", va="center", ha="left")
        ax.text(0.55, 0.28, "p ＜ 0.01", fontsize=10, color="#000000", va="center", ha="left")
        ax.text(0.5, 0.1, "p值", fontsize=10, color="#000000", va="center", ha="center")
        # legend 2
        ax = fig.add_axes([0.7, 0.54, 0.19, 0.15])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axes.set_axis_off()
        ax.add_patch(plt.Rectangle((0.00, 0.01), 0.99, 0.99, edgecolor = [0, 0, 0, 1], facecolor = [1, 1, 1, 1], fill = True, linewidth = 1))
        ax.add_patch(plt.Circle((0.15, 0.85), 0.02, color = "#000000"))
        ax.add_patch(plt.Circle((0.15, 0.60), 0.05, color = "#000000"))
        ax.add_patch(plt.Circle((0.15, 0.35), 0.08, color = "#000000"))
        ax.text(0.25, 0.85, "|cohen'd| ＜ 0.5", fontsize=9, color="#000000", va="center", ha="left")
        ax.text(0.25, 0.60, "|cohen'd| ≥ 0.5", fontsize=9, color="#000000", va="center", ha="left")
        ax.text(0.25, 0.35, "|cohen'd| ≥ 0.8", fontsize=9, color="#000000", va="center", ha="left")
        ax.text(0.5, 0.1, "cohen'd", fontsize=10, color="#000000", va="center", ha="center")
        fig.savefig("work_dirs/study3/aov.jpg", bbox_inches = "tight")

    @staticmethod
    def cohen_d(x: np.array, y: np.array):
        nx, ny = len(x), len(y)
        x_mean_square = (nx - 1) * np.var(x, ddof=0)
        y_mean_square = (ny - 1) * np.var(y, ddof=0)
        return (np.mean(x) - np.mean(y)) / np.sqrt((x_mean_square + y_mean_square) / (nx + ny - 2))


    def step1(self):
        # 获取差异最小的图片，也就是模型识别效果不好的
        torch.cuda.set_device(2)
        train_set = CFDDataset(col=self.col, split=False)
        train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=False)

        backbone: torch.nn.Module = torch.load(self.model_path)
        backbone.cuda()
        backbone.eval()

        embeddings = []
        labels = []
        with torch.no_grad():
            for img, local_labels in tqdm(train_loader):
                local_embeddings: torch.Tensor = backbone(img.cuda())
                embeddings.append(local_embeddings.cpu())
                labels.append(local_labels.cpu())
        embeddings: torch.Tensor = torch.concat(embeddings, dim = 0)
        labels: torch.Tensor = torch.concat(labels, dim = 0)

        feature_class = torch.concat([embeddings[labels == 0].mean(dim = 0).unsqueeze(0),
                                      embeddings[labels == 1].mean(dim = 0).unsqueeze(0)], dim = 0)
        torch.save(feature_class, self.model_save_feature)

        coss = []
        for i in range(embeddings.shape[0]):
            cos = torch.cosine_similarity(embeddings[i], feature_class, dim = 1).unsqueeze(0)
            coss.append(cos)
        coss = torch.concat(coss, dim = 0)
        # 得到模型识别效果不好的id索引 以及文件名称
        return coss.max(dim = 1).values.argmin(), \
               train_set._df["filename"][train_set._idx[int(coss.max(dim = 1).values.argmin())]]

    @torch.no_grad()
    def step2(self, filename):
        # 获取符合要求的随机噪音
        torch.cuda.set_device(2)
        backbone: torch.nn.Module = torch.load(self.model_path)
        backbone.cuda()
        backbone.eval()
        features = torch.load(self.model_save_feature)

        get_embed = lambda img_: backbone(torch.tensor(img_.transpose([2,0,1]), dtype=torch.float32).unsqueeze(0).cuda()).cpu()
        get_cosine = lambda embed_: torch.cosine_similarity(embed_, features, dim = 1)

        im = cv2.imread("../../datasets/1/face/{}".format(filename), cv2.IMREAD_COLOR) # bgr
        im: np.ndarray = ((im / 255) - 0.5) / 0.5

        raw_cos_sim = get_cosine(get_embed(im))
        more = [] # 更像
        less = []
        for _ in tqdm(range(10000)):
            noise = RCIC.general_noise_stand(112)

            img_add = np.zeros(im.shape)
            img_reduce = np.zeros(im.shape)
            img_add[:,:,0] = np.mean([im[:,:,0], noise], axis = 0)
            img_add[:,:,1] = np.mean([im[:,:,1], noise], axis = 0)
            img_add[:,:,2] = np.mean([im[:,:,2], noise], axis = 0)
            img_reduce[:,:,0] = np.mean([im[:,:,0], -noise], axis = 0)
            img_reduce[:,:,1] = np.mean([im[:,:,1], -noise], axis = 0)
            img_reduce[:,:,2] = np.mean([im[:,:,2], -noise], axis = 0)

            noise_cos_sim_add = get_cosine(get_embed(img_add))
            noise_cos_sim_reduce = get_cosine(get_embed(img_reduce))
            if noise_cos_sim_add[1] > noise_cos_sim_reduce[1]:
                more.append(noise)
            else:
                less.append(noise)
        np.save(f"{self.work_path}/more.npy", np.array(more))
        np.save(f"{self.work_path}/less.npy", np.array(less))

    def step3(self, filename):
        im = cv2.imread("../../datasets/1/face/{}".format(filename), cv2.IMREAD_COLOR) # bgr
        im: np.ndarray = ((im / 255) - 0.5) / 0.5

        noise = RCIC.normalized_noise(np.load(f"{self.work_path}/less.npy").mean(axis = 0))
        img_add = np.zeros(im.shape)
        img_reduce = np.zeros(im.shape)
        img_add[:,:,0] = np.mean([im[:,:,0], noise], axis = 0)
        img_add[:,:,1] = np.mean([im[:,:,1], noise], axis = 0)
        img_add[:,:,2] = np.mean([im[:,:,2], noise], axis = 0)
        img_reduce[:,:,0] = np.mean([im[:,:,0], -noise], axis = 0)
        img_reduce[:,:,1] = np.mean([im[:,:,1], -noise], axis = 0)
        img_reduce[:,:,2] = np.mean([im[:,:,2], -noise], axis = 0)

        plt.figure()
        plt.imshow(img_add[:,:,[2,1,0]] / 2 + 0.5)
        plt.axis("off")
        plt.savefig(f"{self.work_path}/add.jpg", bbox_inches = "tight")

        plt.figure()
        plt.imshow(img_reduce[:,:,[2,1,0]] / 2 + 0.5)
        plt.axis("off")
        plt.savefig(f"{self.work_path}/reduce.jpg", bbox_inches = "tight")

    def step4(self, filename):
        im = cv2.imread("../../datasets/1/face/{}".format(filename), cv2.IMREAD_COLOR) # bgr
        im: np.ndarray = ((im / 255) - 0.5) / 0.5

        cmap1 = ListedColormap([[0,0,0,1],
                            [1,0,0,1]])
        cmap2 = ListedColormap([[0,0,0,1],
                            [0,1,0,1]])

        z1,c1,n1 = RCIC.calc_noise_cluster(np.load(f"{self.work_path}/more.npy").mean(axis = 0))
        z2,c2,n2 = RCIC.calc_noise_cluster(np.load(f"{self.work_path}/less.npy").mean(axis = 0))

        plt.figure()
        plt.axis("off")
        plt.imshow((im[:,:,[2,1,0]] + 1) / 2)
        plt.imshow(c1, cmap = cmap1, alpha=0.2)
        plt.imshow(c2, cmap = cmap2, alpha=0.2)
        plt.savefig(f"{self.work_path}/cluster.jpg", bbox_inches = "tight")

if __name__ == "__main__":
    df_v = pd.read_csv("train_a_predict_val.csv")
    df = pd.read_csv("work_dirs/every_eval_the_best_model.csv")
    for i, row in df.iterrows():
        s = study3(df_v.iloc[i, 0], "/".join(row.map(str)))
        s.start()
        del s