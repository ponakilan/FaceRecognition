import torch
from torch import nn
import numpy as np

from .utils.detect import detect_face
from networks.mtcnn import PNet, RNet, ONet


class MTCNN(nn.Module):
    def __init__(
        self, min_face_size=20,
        thresholds=(0.6, 0.7, 0.7), factor=0.709,
        select_largest=True, weights_path=None,
        transform=None, device=None
    ):
        super().__init__()

        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.select_largest = select_largest
        self.transform = transform

        self.pnet = PNet(pretrained=True, weights_path=weights_path)
        self.rnet = RNet(pretrained=True, weights_path=weights_path)
        self.onet = ONet(pretrained=True, weights_path=weights_path)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def postprocess_faces(self, img, batch_boxes, batch_probs):
        faces = []
        face_tensors = []
        if self.transform and batch_probs is not None:
            for i, box in enumerate(batch_boxes):
                if batch_probs[i] >= 0.9:
                    face = img.crop(box)
                    faces.append(face)
                    face = self.transform(face)
                    face_tensors.append(face)
        return faces, face_tensors

    def forward(self, img):
        batch_boxes, batch_probs = self.detect(img)
        # return self.postprocess_faces(img, batch_boxes, batch_probs)
        return batch_boxes, batch_probs

    def detect(self, img):

        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img, self.min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor,
                self.device
            )

        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
        boxes = np.array(boxes, dtype=object)
        probs = np.array(probs, dtype=object)

        if (
            not isinstance(img, (list, tuple)) and
            not (isinstance(img, np.ndarray) and len(img.shape) == 4) and
            not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
        ):
            boxes = boxes[0]
            probs = probs[0]

        return boxes, probs
