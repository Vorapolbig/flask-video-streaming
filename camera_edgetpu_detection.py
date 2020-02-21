# Copyright 2019, 2020 NXP
# SPDX-License-Identifier: MIT

# Based on example code from:
#  flask-video-streaming/camera_opencv.py, and
#  edgetpu/edgetpu/demo

import cv2 as cv
from base_camera import BaseCamera

import numpy as np
import time
import os

from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.detection.engine import DetectionEngine
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

if os.environ.get('MODEL'):
  model = os.environ.get('MODEL')
else:
  model = '../model/coral_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'

if os.environ.get('LABELS'):
  labelsPath = os.environ.get('LABELS')
else:
  labelsPath = '../model/coral_models/coco_labels.txt'

if os.environ.get('DETECTED_ID'):
  detected_id = os.environ.get('DETECTED_ID')
else:
  detected_id = 0

# Function to read labels from text files.
def ReadLabelFile(file_path):
  with open(file_path, 'r', encoding="utf-8") as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret

labels = ReadLabelFile(labelsPath)

# RGB
classColors = np.array(((210, 230, 23), (255, 165, 0), (192, 192, 192)))
if len(labels) > 3:
    classColors = np.append(classColors, np.random.randint(0, 255, size=(len(labels)-3, 3), dtype="uint8"), axis=0)

# handle case where index is out of bounds
labels[len(labels)] = 'Unknown'
classColors = np.append(classColors, [[0,0,0]], axis=0)

if os.environ.get('THRESHOLD'):
    confThreshold = float(os.environ.get('THRESHOLD'))
else:
    confThreshold = 0.5
nmsThreshold = 0.4

if os.environ.get('FPS') and os.environ.get('FPS') == "True":
    showfps = True
    tickfreq = cv.getTickFrequency()
else:
    showfps = False

font = ImageFont.truetype('FreeMono.ttf', 30)
person_cnt = 0

# edgetpu = '/dev/apex_0'
# edgetpu = '/dev/apex_1'
# edgetpu = '/sys/bus/usb/devices/2-1.3'
if os.environ.get('EDGETPU'):
  edgetpu = os.environ.get('EDGETPU')
  print('Opening ', model, edgetpu)
  engine = DetectionEngine(model, edgetpu)
else:
  print('Opening', model)
  engine = DetectionEngine(model)

class Camera(BaseCamera):
    video_source = 0
    start = cv.getTickCount()

    @staticmethod
    def set_video_source(source):
        try:
            Camera.video_source = int(source)
        except ValueError:
            Camera.video_source = source

    @staticmethod
    def frames():
        global showfps
        global person_cnt

        print("Camera.video_source={}".format(Camera.video_source))
        camera = cv.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

#           rows = img.shape[0]
#           cols = img.shape[1]

            img_pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#            objects = engine.DetectWithImage(img_pil, threshold=confThreshold, keep_aspect_ratio=True, relative_coord=False, top_k=10)
            objects = engine.detect_with_image(img_pil, threshold=confThreshold, keep_aspect_ratio=True, relative_coord=False, top_k=10)
            detected_cnt = 0
            draw = ImageDraw.Draw(img_pil)
            for obj in objects:
                box = obj.bounding_box.flatten().tolist()
                id = obj.label_id
#               print(id, obj.score, box)
                if id >= len(labels):
                    id = len(labels)-1
                color = tuple(classColors[id])
                draw.rectangle(box, outline=color, width=4)
                if id == detected_id:
                    detected_cnt += 1
                    text = "{}: {:d} {:.2f}".format(labels[id], detected_cnt, obj.score)
                else:
                    text = "{}: {:.2f}".format(labels[id], obj.score)
                draw.text((box[0], box[1]-5), text, font=font, fill=color)

            title = ''
            if showfps:
                now = cv.getTickCount()
                fps = tickfreq / (now - Camera.start)
                Camera.start = now
                title = "FPS: {:5,.0f}, Detected: {:<3d}".format(round(fps,0), detected_cnt)
            elif detected_cnt > 0:
                title = "Detected: {:<3d}".format(detected_cnt)
            draw.text((20,20), title, font=font, fill=(250,250,250))

            # encode as a jpeg image and return it
            wf = BytesIO()
            img_pil.save(wf, format='jpeg')
            yield wf.getbuffer()
