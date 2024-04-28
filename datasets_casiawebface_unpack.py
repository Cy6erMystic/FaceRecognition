import os
import cv2
from multiprocessing import Manager, Queue, Pool

from datasets.dataset_mx import MXFaceDataset
from configs import getLogger

logger = getLogger("datasets")
def parse_index(q: Queue):
    mx = MXFaceDataset("../../datasets/faces_webface_112x112")
    logger.info("加载数据完成")
    for i in mx.imgidx[239000:]:
        q.put(i)
    q.put(None)

def parse_img(m: Queue, c: Queue):
    mx = MXFaceDataset("../../datasets/faces_webface_112x112")
    logger.info("加载数据完成")
    while True:
        i = m.get()
        if i is None:
            m.put(None)
            c.put(None, None, None, None)
            break
        header, img = mx.unpack(i)
        if os.path.exists("../../datasets/1/face/2_{}_{}.jpg".format(header.label, i)):
            logger.info("存在: {}".format(i))
            continue
        c.put((header.label, i, img, len(mx.imgidx)))
        logger.info("加载: {} len: {}".format(i, c.qsize()))

def save_img(c: Queue):
    while True:
        header, i, img, l = c.get()
        if i is None:
            c.put(None, None, None, None)
            break
        cv2.imwrite("../../datasets/1/face/2_{}_{}.jpg".format(header, i), img)
        logger.info("完成: {}/{}".format(i, l))

if __name__ == "__main__":
    with Manager() as ma:
        if not os.path.exists("../../datasets/1/face"):
            os.mkdir("../../datasets/1/face")
        qi = ma.Queue(maxsize=20)
        qf = ma.Queue(maxsize=20)
        with Pool(processes=16) as p:
            p.apply_async(func=parse_index, args=(qi,))
            p.apply_async(func=parse_img, args=(qi,qf))
            p.apply_async(func=parse_img, args=(qi,qf))
            p.apply_async(func=parse_img, args=(qi,qf))
            p.apply_async(func=parse_img, args=(qi,qf))
            p.apply_async(func=save_img, args=(qf,))
            p.apply_async(func=save_img, args=(qf,))
            p.apply_async(func=save_img, args=(qf,))
            p.apply_async(func=save_img, args=(qf,))
            p.apply_async(func=save_img, args=(qf,))
            p.apply_async(func=save_img, args=(qf,))
            p.apply_async(func=save_img, args=(qf,))
            p.apply_async(func=save_img, args=(qf,))
            p.apply_async(func=save_img, args=(qf,))
            p.close()
            p.join()
