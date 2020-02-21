# Copyright 2020 NXP
# SPDX-License-Identifier: MIT

# Based on example code from:
#  flask-video-streaming/camera_opencv.py, and
#  google-coral/project-posenet/pose_camera.py

import cv2 as cv
from base_camera import BaseCamera

import numpy as np
import time
import os

#from edgetpu.basic.basic_engine import BasicEngine
from pose_engine import PoseEngine
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

font = ImageFont.truetype('FreeMono.ttf', 30)

# from pose_camera import EDGES, shadow_text, draw_pose
EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)


def shadow_text(dwg, x, y, text):
    dwg.text((x + 1, y + 1), text, font=font, fill='black')
    dwg.text((x, y),         text, font=font, fill='white')

def draw_pose(dwg, pose, img_size_wxh, color='yellow', threshold=0.2):
#def draw_pose(dwg, pose, img_size, inference_box, color='yellow', threshold=0.2):
#   box_x, box_y, box_w, box_h = inference_box
#   scale_x, scale_y = img_size[0] / box_w, img_size[1] / box_h
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        # Offset and scale to source coordinate space.
#        kp_y = int((keypoint.yx[0] - box_y) * scale_y)
#        kp_x = int((keypoint.yx[1] - box_x) * scale_x)
        kp_y = int(keypoint.yx[0])
        kp_x = int(keypoint.yx[1])

        xys[label] = (kp_x, kp_y)
        dwg.arc( [ (int(kp_x-2.5), int(kp_y-2.5)), (int(kp_x+2.5), int(kp_y+2.5))], 0, 360,
                           fill=color)

    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        dwg.line([(ax, ay), (bx, by)], fill=color, width=2)

# end pose_camera.py

if os.environ.get('MODEL'):
  model = os.environ.get('MODEL')
else:
  model = '../model/posenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'

if os.environ.get('FPS') and os.environ.get('FPS') == "True":
    showfps = True
    tickfreq = cv.getTickFrequency()
else:
    showfps = False

font = ImageFont.truetype('FreeMono.ttf', 30)

# edgetpu = '/dev/apex_0'
# edgetpu = '/dev/apex_1'
# edgetpu = '/sys/bus/usb/devices/2-1.3'
if os.environ.get('EDGETPU'):
  edgetpu = os.environ.get('EDGETPU')
  print('Opening ', model, ' ', edgetpu)
  engine = PoseEngine(model, edgetpu)
else:
  print('Opening ', model)
  engine = PoseEngine(model)

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

        print("Camera.video_source={}".format(Camera.video_source))
        camera = cv.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        _, img = camera.read()
        img_size_wxh = (img.shape[1], img.shape[0])
        print(img_size_wxh)

        input_shape = engine.get_input_tensor_shape()
        inference_size = (input_shape[2], input_shape[1])
        print(inference_size)

        while True:
            # read current frame
            _, img = camera.read()

#           DetectPosesInImages takes int8 numpy object in [Y,X,RGB] format.
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            poses, interence_time = engine.DetectPosesInImage(img_rgb)

            img_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil)

#            text_line = 'PoseNet: Nposes %d' % (len(poses))
#            shadow_text(draw, 10, 20, text_line)

            for pose in poses:
                draw_pose(draw, pose, img_size_wxh)

            # encode as a jpeg image and return it
            wf = BytesIO()
            img_pil.save(wf, format='jpeg')
            yield wf.getbuffer()
