from model import RetianFaceDetection, MogFaceDetaction

if __name__ == "__main__":
    fd = RetianFaceDetection(pretain_model="../pretrain/RetinaFace-R50/Resnet50_Final.pth", device="cuda:1")
    bboxs = fd.calc_bbox("work_dirs/face_detection/t6.jpg")
    fd.render_bbox(bboxs, "work_dirs/face_detection/t6.jpg")

    fd = MogFaceDetaction(pretain_model = "../pretrain/MogFace/model_140000.pth", device="cuda:0")
    bboxes = fd.calc_bbox("work_dirs/face_detection/t6.jpg")
    fd.render_bbox(bboxes, "work_dirs/face_detection/t6.jpg")