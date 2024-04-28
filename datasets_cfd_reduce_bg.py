import os
import cv2
from multiprocessing import Manager, Queue, Pool

from model import RetianFaceDetection, MogFaceDetaction
from configs import getLogger

logger = getLogger("datasets")
def list_img_path(path: str, img_q: Queue, parent = True):
    for item in os.listdir(path):
        path_ = os.path.join(path, item)
        if os.path.isdir(path_):
            list_img_path(path_, img_q, False)
        elif os.path.isfile(path_):
            if os.path.basename(path_).startswith("CFD"):
                im = cv2.imread(path_, cv2.IMREAD_COLOR)
                logger.info("加载图片: {}".format(path_))
                img_q.put((os.path.basename(path_), im))
    if parent:
        img_q.put((None, None))

def parse_img_face(img_q: Queue, device: str = "cuda:0"):
    if not os.path.exists("work_dirs/CFD"):
        os.mkdir("work_dirs/CFD")
        os.mkdir("work_dirs/CFD/r")
        os.mkdir("work_dirs/CFD/s")
    logger.info("等待模型加载")
    fd = MogFaceDetaction(pretain_model = "../pretrain/MogFace/model_140000.pth", device=device)
    while True:
        logger.info("等待新的图片")
        file_name, img_content = img_q.get()
        logger.info("解析器拿到图片: {}".format(file_name))
        if img_content is None:
            img_q.put((None, None))
            break
        img_content = cv2.resize(img_content, None, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
        logger.info("解析图片: {}".format(file_name))
        bboxs = fd.calc_bbox(img_content)
        for i, im in enumerate(fd.split_imgs(bboxs, img_content, False)):
            cv2.imwrite("work_dirs/CFD/r/{}_{}".format(i, file_name),
                        im)
        for i, im in enumerate(fd.split_imgs(bboxs, img_content, True)):
            # cv2.imwrite("work_dirs/CFD/s/{}_{}".format(i, file_name),
            #             im)
            cv2.imwrite("../../datasets/1/face/{}_{}".format(i, file_name),
                        im)

if __name__ == "__main__":
    # 解析文件
    logger.info("开始转换程序")
    with Manager() as manager:
        img_queue = manager.Queue(maxsize=20)
        logger.info("初始化进程管理器: 成功")

        with Pool(processes=10) as p:
            p.apply_async(func=list_img_path,
                          args=("/media/s5t/datasets/CFD Version 3.0/Images",
                                img_queue))
            p.apply_async(func=parse_img_face,
                          args=(img_queue,
                                "cuda:1"))
            p.apply_async(func=parse_img_face,
                          args=(img_queue,
                                "cuda:1"))
            p.apply_async(func=parse_img_face,
                          args=(img_queue,
                                "cuda:1"))
            p.apply_async(func=parse_img_face,
                          args=(img_queue,
                                "cuda:1"))
            p.apply_async(func=parse_img_face,
                          args=(img_queue,
                                "cuda:1"))
            p.close()
            p.join()