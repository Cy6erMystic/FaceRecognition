import os
import cv2
from multiprocessing import Manager, Queue, Pool

from datasets.dataset_mx import MXFaceDataset
from model import RetianFaceDetection, MogFaceDetaction
from configs import getLogger

logger = getLogger("datasets")
def parse_img(mx: MXFaceDataset, i: int):
    header, img = mx.unpack(i)
    if os.path.exists("work_dirs/glint/{}_{}.jpg".format(header.label[0], i)):
        logger.info("存在: {}".format(i))
        return 0
    img = cv2.resize(img, None, None, fx=256 / 112, fy=256 / 112, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("work_dirs/glint/{}_{}.jpg".format(header.label[0], i), img)
    # open("work_dirs/glint/{}_{}.jpg".format(header.label[0], i), "wb").write(img)
    logger.info("完成: {}/{}".format(i, len(mx.imgidx)))

if __name__ == "__main__":
    mx = MXFaceDataset("/media/s5t/caai2024/datasets/glint360k")
    if not os.path.exists("work_dirs/glint"):
        os.mkdir("work_dirs/glint")
    logger.info("加载数据库完成")
    with Pool(processes=8) as p:
        for i in mx.imgidx:
            p.apply_async(parse_img, args=(mx, i))
        p.close()
        p.join()