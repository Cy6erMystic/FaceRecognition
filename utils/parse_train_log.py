import os
import re

def get_files(path: str):
    return os.listdir(path)

def get_data_from_file(path: str):
    best_acc = 0
    with open(path, "r") as f:
        file_contents = f.readlines()
        file_contents.reverse()
        for fc in file_contents:
            if fc.find("BEST_ACC") >= 0:
                best_acc = float(re.findall(r"BEST_ACC:\s(.*?)\n", fc)[0])
                break
    _, col_name, model_name, param1, param2, param3 = os.path.basename(path).replace(".logs", "").split("_")
    return col_name, model_name, int(param1), float(param2), float(param3), best_acc
