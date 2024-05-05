from __future__ import annotations
import os
import datetime
from configs.base import ModelChoose

class Logger():
    def __init__(self, name: str, mmc: ModelChoose) -> None:
        self._name = name
        self._file_path = "logs/{}_{}_{}_{}_{}_{}.logs".format(
            self._name,
            mmc.col_name,
            mmc.model_name,
            mmc.param1,
            mmc.param2,
            mmc.param3
        )
        if not os.path.exists("logs"):
            os.mkdir("logs")
        self._file = open(self._file_path, "w", encoding="utf-8")

    def __del__(self):
        self._file.close()

    @property
    def curr_time(self):
        # 北京 UTC +8
        return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")

    def _p(self, msg: str):
        print(msg)
        self._file.write(self.curr_time)
        self._file.write(" ")
        self._file.write(msg)
        self._file.write("\n")
        self._file.flush()

    def info(self, *msgs: list[str]):
        self._p(", ".join(map(str, msgs)))
