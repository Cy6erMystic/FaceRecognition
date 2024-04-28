from __future__ import annotations
import os
import cv2
import copy
import torch
import numpy as np
from torch import distributed

from backbone.retinaface import RetinaFace
from utils.face_detection import PriorBox

from backbone.mogface import ResNet, LFPN, MogPredNet, WiderFaceBaseNet
from utils.face_detection import GeneartePriorBoxes

def init_distributed(rank: int, world_size: int, addr: str = "tcp://127.0.0.1:12586"):
    distributed.init_process_group(
        backend="nccl",
        init_method=addr,
        rank=rank,
        world_size=world_size,
    )

class BaseFaceDetection():
    VIS_THRES = 0.5
    TARGET_SHAPE = (112, 112) # H, W
    @classmethod
    def _reshape_img(cls, img: np.ndarray) -> np.ndarray:
        resize = min(cls.TARGET_SHAPE[1] / img.shape[1], cls.TARGET_SHAPE[0] / img.shape[0])
        if resize != 1:
            img: np.ndarray = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        return img

    @classmethod
    def _reshape_bbox(cls, bbox: np.ndarray):
        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]
        if h > w:
            l = h - w
            bbox[0] -= l // 2
            bbox[2] += l - (l // 2)
        else:
            l = w - h
            bbox[1] -= l // 2
            bbox[3] += l - (l // 2)
        return bbox

    def split_imgs(self, bboxs: list, img_raw: np.ndarray, square = False):
        img_raw = copy.deepcopy(img_raw)
        imgs = []
        for b in bboxs:
            if b[4] < self.VIS_THRES:
                continue
            b = list(map(int, b))
            if square:
                b = self._reshape_bbox(b)
            img_ = self._reshape_img(img_raw[b[1]:b[3], b[0]:b[2]])
            imgs.append(img_)
        return imgs

class RetianFaceDetection(BaseFaceDetection):
    confidence_threshold = 0.02
    nms_threshold = 0.4
    target_size = 1600
    max_size = 2150
    def __init__(self, pretain_model: str = "", device = "cuda:0") -> None:
        self.cfg = {
            'name': 'Resnet50',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True,
            'batch_size': 24,
            'ngpu': 4,
            'epoch': 100,
            'decay1': 70,
            'decay2': 90,
            'image_size': 840,
            'pretrain': True,
            'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
            'in_channel': 256,
            'out_channel': 256
        }
        self.net = self._init_model(pretain_model)
        self.device = device
        self.net.to(self.device)

    def _init_model(self, pretain_model: str):
        net = RetinaFace(self.cfg, 'test')
        if isinstance(pretain_model, str) and len(pretain_model) > 0 and os.path.exists(pretain_model):
            pretrained_dict = torch.load(pretain_model)
        f = lambda x: x.split('module.', 1)[-1] if x.startswith('module.') else x
        pretrained_dict = {f(key): value for key, value in pretrained_dict.items()}

        net.load_state_dict(pretrained_dict, strict=False)
        net.eval()
        return net

    @staticmethod
    def _decode(loc, priors, variances):
        # Adapted from https://github.com/Hakuyume/chainer-ssd
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """

        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @staticmethod
    def _decode_landm(pre, priors, variances):
        """Decode landm from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            pre (tensor): landm predictions for loc layers,
                Shape: [num_priors,10]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded landm predictions
        """
        landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), dim=1)
        return landms

    def _resize_img(self, img: np.ndarray):
        im_shape = img.shape

        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(self.target_size) / float(im_size_min)
        if np.round(resize * im_size_max) > self.max_size:
            resize = float(self.max_size) / float(im_size_max)
        if resize != 1:
            img: np.float32 = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        return img, resize

    @torch.no_grad()
    def calc_bbox(self, img_raw):
        img, resize = self._resize_img(np.float32(img_raw))

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img: torch.Tensor = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        output: tuple[torch.Tensor] = self.net(img)
        loc = output[0]
        conf = output[1]
        landms = output[2]

        priorbox = PriorBox(self.cfg, image_size = (im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data

        boxes = self._decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores: np = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = self._decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:10, :]
        # landms = landms[:10, :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets

    def render_bbox(self, bboxs, img_raw):
        img_raw = copy.deepcopy(img_raw)
        for b in bboxs:
            if b[4] < self.VIS_THRES:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        return img_raw

class MogFaceDetaction(BaseFaceDetection):
    pre_nms_top_k = 5000
    max_bbox_per_img = 750
    nms_th = 0.3
    score_th = 0.01
    def __init__(self, pretain_model: str = "", device = "cuda:2") -> None:
        self.net: WiderFaceBaseNet = self._init_model()
        self.net.to(device)
        if isinstance(pretain_model, str) and os.path.exists(pretain_model):
            self.net.load_state_dict(torch.load(pretain_model, map_location = device))
        self.device = device

    def _init_model(self) -> WiderFaceBaseNet:
        self._priorbox = GeneartePriorBoxes(scale_list=[0.68],
                                            aspect_ratio_list=[1.0],
                                            stride_list=[4,8,16,32,64,128],
                                            anchor_size_list=[16,32,64,128,256,512])
        return WiderFaceBaseNet(backbone=ResNet(depth=101),
                                fpn = LFPN(c2_out_ch=256,
                                           c3_out_ch=512,
                                           c4_out_ch=1024,
                                           c5_out_ch=2048,
                                           c6_mid_ch=512,
                                           c6_out_ch=512,
                                           c7_mid_ch=128,
                                           c7_out_ch=256,
                                           out_dsfd_ft=True),
                                pred_net = MogPredNet(num_anchor_per_pixel=1,
                                                      input_ch_list=[256,256,256,256,256,256],
                                                      use_deep_head=True,
                                                      deep_head_with_gn=True,
                                                      deep_head_ch=512,
                                                      use_ssh=True,
                                                      use_dsfd=True,
                                                      num_classes=1,
                                                      phase="test"),
                                phase="test")

    @staticmethod
    def _decode(loc, anchors):
        """
        loc: torch.Tensor
        anchors: 2-d, torch.Tensor (cx, cy, w, h)
        boxes: 2-d, torch.Tensor (x0, y0, x1, y1)
        """
        boxes = torch.cat((
            anchors[:, :2] + loc[:, :2] * anchors[:, 2:],
            anchors[:, 2:] * torch.exp(loc[:, 2:])), 1)

        boxes[:, 0] -= (boxes[:,2] - 1 ) / 2
        boxes[:, 1] -= (boxes[:,3] - 1 ) / 2
        boxes[:, 2] += boxes[:,0] - 1  
        boxes[:, 3] += boxes[:,1] - 1 

        return boxes

    @staticmethod
    def _transform_anchor(anchors):
        '''
        from [x0, x1, y0, y1] to [c_x, cy, w, h]
        x1 = x0 + w - 1
        c_x = (x0 + x1) / 2 = (2x0 + w - 1) / 2 = x0 + (w - 1) / 2
        '''
        return np.concatenate(((anchors[:, :2] + anchors[:, 2:]) / 2 , anchors[:, 2:] - anchors[:, :2] + 1), axis=1)

    @torch.no_grad()
    def calc_bbox(self, img_raw):
        img = np.float32(img_raw)
        max_im_shrink = (0x7fffffff / 200.0 / (img.shape[0] * img.shape[1])) ** 0.5
        max_im_shrink = 2.2 if max_im_shrink > 2.2 else max_im_shrink

        shrink = max_im_shrink if max_im_shrink < 1 else 1
        if shrink != 1:
            img = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
        width = img.shape[1]
        height = img.shape[0]

        x = torch.from_numpy(img).permute(2, 0, 1)
        x = x.unsqueeze(0)
        out: torch.Tensor = self.net(x.to(self.device))

        anchors = self._transform_anchor(self._priorbox(height, width))
        anchors = torch.FloatTensor(anchors).to(self.device)
        decode_bbox =  self._decode(out[1].squeeze(0), anchors)
        boxes = decode_bbox
        scores = out[0].squeeze(0)

        top_k = self.pre_nms_top_k
        _, idx = scores[:, 0].sort(0)
        idx = idx[-top_k:]
        boxes = boxes[idx]
        scores = scores[idx]

        # [11620, 4]
        boxes = boxes.cpu().numpy()
        w = boxes[ :, 2] - boxes[:,0] + 1
        h = boxes[ :, 3] - boxes[:,1] + 1
        boxes[:,0] /= shrink
        boxes[:,1] /= shrink
        boxes[:,2] = boxes[:,0] + w / shrink - 1
        boxes[:,3] = boxes[:,1] + h / shrink - 1
        #boxes = boxes / shrink
        # [11620, 2]
        scores = scores.cpu().numpy()

        inds = np.where(scores[:, 0] > self.score_th)[0]
        if len(inds) == 0:
            det = np.empty([0, 5], dtype=np.float32)
            return det
        c_bboxes = boxes[inds]
        # [5,]
        c_scores = scores[inds, 0]
        # [5, 5]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
        
        #starttime = datetime.datetime.now()
        keep = py_cpu_nms(c_dets, self.nms_th)
        #endtime = datetime.datetime.now()
        #print('nms forward time = ',(endtime - starttime).seconds+(endtime - starttime).microseconds/1000000.0,' s')
        c_dets = c_dets[keep, :]

        max_bbox_per_img = self.max_bbox_per_img
        if max_bbox_per_img > 0:
            image_scores = c_dets[:, -1]
            if len(image_scores) > max_bbox_per_img:
                image_thresh = np.sort(image_scores)[-max_bbox_per_img]
                keep = np.where(c_dets[:, -1] >= image_thresh)[0]
                c_dets = c_dets[keep, :]
        return c_dets

    def render_bbox(self, bboxs, img_raw: np.ndarray):
        img_raw = copy.deepcopy(img_raw)
        for b in bboxs:
            if b[4] < self.VIS_THRES:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        return img_raw

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores: np = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order: np = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
