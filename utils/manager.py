import numpy as np

class ModelManager():
    """在模型训练的时候, 需要保存的数据, 包括以下内容:
    - 损失值
    - 准确率
    """
    def __init__(self) -> None:
        self.reset_loss()
        self.reset_acc()

    def set_epoch(self, epoch: int):
        self._curr_epoch = epoch

    def update_loss(self, val):
        self.loss_list.append((self._curr_epoch, val))

    def update_acc(self, acc: float, std: float):
        self.acc_list.append((self._curr_epoch, acc, std))

    def reset_loss(self):
        self.loss_list = []

    def reset_acc(self):
        self.acc_list = []

    @property
    def is_best_acc(self):
        i = np.argmax([a[1] for a in self.acc_list])
        return self._curr_epoch == self.acc_list[i][0]

    def __str__(self) -> str:
        r1 = []
        for loss in self.loss_list:
            if loss[0] == self._curr_epoch:
                r1.append(loss[1])

        r2 = []
        for acc in self.acc_list:
            if acc[0] == self._curr_epoch:
                r2.append(acc[1])

        msgs = ["EPOCH: %s" % (self._curr_epoch),
                "LOSS: %.4f" % (np.mean(r1)),
                "ACC: %1.5f" % (np.mean(r2)),
                "BSET_ACC: %1.5f" % (np.max([a[1] for a in self.acc_list]))]
        return " ".join(msgs)