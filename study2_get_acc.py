import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.parse_train_log import get_files, get_data_from_file

a = []
for path in get_files("logs"):
    a.append(get_data_from_file(os.path.join("logs", path)))
df = pd.DataFrame(a, columns=["col", "model", "p1", "p2", "p3", "acc"])
df["m"] = df["model"] + "-" + df["p1"].map(str) + "-" + df["p2"].map(str) + "-" + df["p3"].map(str)
df.pivot_table(
    values="acc",
    index="col",
    columns=["model", "p1", "p2", "p3"]
).iloc[:, [9,3,4,5,6,7,8,0,1,2]]

margin = 0.1
col_to_name_cn = {'S001': '种族', 'S002': '性别', 'R002': '年龄', 'R003': '女性概率', 'R004': '男性概率', 'R006': '黑人概率', 'R010': '白人概率', 'R011': '害怕', 'R012': '生气', 'R013': '吸引力', 'R014': '娃娃脸', 'R015': '厌恶', 'R016': '主导', 'R017': '女性化', 'R018': '快乐', 'R019': '男性化', 'R020': '典型种族人', 'R021': '伤心', 'R023': '惊讶', 'R024': '威胁', 'R025': '值得信赖', 'R026': '不同寻常', 'P001': '亮度中值', 'P002': '鼻宽', 'P003': '鼻长', 'P004': '唇厚', 'P005': '面长', 'P006': '眼高R', 'P007': '眼高L', 'P008': '平均眼高', 'P009': '眼宽R', 'P010': '眼宽L', 'P011': '平均眼宽', 'P012': '脸宽脸颊', 'P013': '脸宽嘴', 'P014': '面宽最大值', 'P015': '前额', 'P017': '上面长度2', 'P018': '瞳孔顶端R', 'P019': '瞳孔顶部L', 'P021': '瞳唇R', 'P022': '瞳唇L', 'P023': '瞳孔唇部平均', 'P026': '下唇下巴', 'P027': '中颊下巴R', 'P028': '中颊下巴L', 'P029': '脸颊平均', 'P030': '中眉发际线R', 'P031': '中眉发际线L', 'P032': '中眉发际线平均', 'P051': '脸型', 'P052': '心形', 'P053': '鼻子形状', 'P054': '嘴唇丰满度', 'P055': '眼型', 'P056': '眼睛大小', 'P057': '上头长', 'P058': '中面部长度', 'P059': '下巴长度', 'P060': '额头高度', 'P061': '颧骨高度', 'P062': '颧骨突出', 'P063': '端面圆度', 'P065': '面部宽高比'}
col_to_name_en = {'S001': 'EthnicitySelf', 'S002': 'GenderSelf', 'R002': 'AgeRated', 'R003': 'FemaleProb', 'R004': 'MaleProb', 'R006': 'BlackProb', 'R010': 'WhiteProb', 'R011': 'Afraid', 'R012': 'Angry', 'R013': 'Attractive', 'R014': 'Babyfaced', 'R015': 'Disgusted', 'R016': 'Dominant', 'R017': 'Feminine', 'R018': 'Happy', 'R019': 'Masculine', 'R020': 'Prototypic', 'R021': 'Sad', 'R023': 'Surprised', 'R024': 'Threatening', 'R025': 'Trustworthy', 'R026': 'Unusual', 'P001': 'LuminanceMedian', 'P002': 'NoseWidth', 'P003': 'NoseLength', 'P004': 'LipThickness', 'P005': 'FaceLength', 'P006': 'EyeHeightR', 'P007': 'EyeHeightL', 'P008': 'EyeHeightAvg', 'P009': 'EyeWidthR', 'P010': 'EyeWidthL', 'P011': 'EyeWidthAvg', 'P012': 'FaceWidthCheeks', 'P013': 'FaceWidthMouth', 'P014': 'FaceWidthBZ', 'P015': 'Forehead', 'P017': 'UpperFaceLength2', 'P018': 'PupilTopR', 'P019': 'PupilTopL', 'P021': 'PupilLipR', 'P022': 'PupilLipL', 'P023': 'PupilLipAvg', 'P026': 'BottomLipChin', 'P027': 'MidcheekChinR', 'P028': 'MidcheekChinL', 'P029': 'CheeksAvg', 'P030': 'MidbrowHairlineR', 'P031': 'MidbrowHairlineL', 'P032': 'MidbrowHairlineAvg', 'P051': 'FaceShape', 'P052': 'Heartshapeness', 'P053': 'NoseShape', 'P054': 'LipFullness', 'P055': 'EyeShape', 'P056': 'EyeSize', 'P057': 'UpperHeadLength', 'P058': 'MidfaceLength', 'P059': 'ChinLength', 'P060': 'ForeheadHeight', 'P061': 'CheekboneHeight', 'P062': 'CheekboneProminence', 'P063': 'FaceRoundness', 'P065': 'fWHR2'}

cols = df["col"].unique()
cols.sort()
models = df["m"].unique()[[0,1,2,9,3,4,5,6,7,8]]

colors = ["#a1b92e", 
          "#66a3d2", "#0b61a4", "#3f92d2",
          "#ffc373", "#ff9200", "#ffad40", 
          "#d25fd2", "#a600a6", "#d235d2"]

plt.figure(figsize=(10,8))
width = (1 - margin) / cols.shape[0]
for i, model in enumerate(models):
    df_t = df.query("m == '{}'".format(model))
    position = [np.where(cols == col)[0][0] + i * width - width * (cols.shape[0] - 1) / 2 for col in df_t["col"]]
    plt.bar(x=position, height=df_t["acc"], width=width,
           label=model, align="center", color = colors[i])
plt.legend(loc="upper left")
plt.ylim(0.45, 1.0)
plt.xticks(ticks=[i for i in range(len(cols))],
           labels=cols, rotation = 20)
        #    labels=[col_to_name_en[col] for col in cols], rotation = 20)
print("")