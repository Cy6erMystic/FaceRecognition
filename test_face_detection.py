import cv2
from model import RetianFaceDetection, MogFaceDetaction

if __name__ == "__main__":
    img_raw = cv2.imread("work_dirs/face_detection/s1.jpg", cv2.IMREAD_COLOR)

    # fd1 = RetianFaceDetection(pretain_model="../pretrain/RetinaFace-R50/Resnet50_Final.pth", device="cuda:1")
    # bboxs1 = fd1.calc_bbox(img_raw)
    # cv2.imwrite("work_dirs/face_detection/retian.jpg", fd1.render_bbox(bboxs1, img_raw))

    img_raw = cv2.resize(img_raw, None, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
    fd2 = MogFaceDetaction(pretain_model = "../pretrain/MogFace/model_140000.pth", device="cuda:1")
    bboxs2 = fd2.calc_bbox(img_raw)
    cv2.imwrite("work_dirs/face_detection/mog.jpg", fd2.render_bbox(bboxs2, img_raw))
    cv2.imwrite("work_dirs/face_detection/split.jpg", fd2.split_imgs(bboxs2, img_raw, True)[0])