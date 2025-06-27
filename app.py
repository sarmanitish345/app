import torch
import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import sys


sys.path.append('./yolov5')
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

device = select_device('')
model = DetectMultiBackend('best.pt', device=device)
model.eval()

def detect_image(img):
    img0 = img.copy()
    img = letterbox(img0, 640, stride=32, auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]  
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    pred = model(img, augment=False)
    pred = non_max_suppression(pred, 0.25, 0.45)[0]

    for *xyxy, conf, cls in pred:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        xyxy = [int(x.item()) for x in xyxy]
        cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
        cv2.putText(img0, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    return img0

gr.Interface(fn=detect_image,
             inputs=gr.Image(type="numpy", label="Upload Image"),
             outputs=gr.Image(label="Prediction"),
             title="YOLOv5 Thermal Detection").launch()
