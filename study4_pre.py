# 预处理tcav数据结构
import cv2, os
from tqdm import tqdm
from datasets.dataset_cfd import CFDDataset
datasets = CFDDataset(split = False)
df = datasets._df.iloc[:, [0, 15] + [i for i in range(24, 66)]]

for i in tqdm(range(df.shape[1] - 1)):
    val_m = df.iloc[:, [i + 1]].mean().item()
    for j, row in df.iloc[:, [0, i + 1]].iterrows():
        filename = row["filename"]
        key = row.keys()[1]
        val = row[key]
        
        for k in ["h", "l"]:
            if not os.path.exists("../../datasets/1/tcav/{}_{}".format(key, k)):
                os.makedirs("../../datasets/1/tcav/{}_{}".format(key, k))

        im = cv2.imread("../../datasets/1/face/{}".format(filename), cv2.IMREAD_COLOR)
        if val > val_m:
            cv2.imwrite("../../datasets/1/tcav/{}_h/{}".format(key, filename), im)
        else:
            cv2.imwrite("../../datasets/1/tcav/{}_l/{}".format(key, filename), im)