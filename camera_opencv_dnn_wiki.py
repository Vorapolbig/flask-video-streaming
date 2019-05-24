# Copyright 2019 NXP
# SPDX-License-Identifier: MIT

# Based on example code from:
#  https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
#  flask-video-streaming/camera_opencv.py

import os
import cv2 as cv
from base_camera import BaseCamera

modelDir='../model'

model = os.path.sep.join([modelDir, "frozen_inference_graph.pb"])
graph = os.path.sep.join([modelDir, "graph.pbtxt"])
if os.environ.get('THRESHOLD'):
    confThreshold = float(os.environ.get('THRESHOLD'))
else:
    confThreshold = 0.5

cvNet = cv.dnn.readNetFromTensorflow(model, graph)

class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        try:
            Camera.video_source = int(source)
        except ValueError:
            Camera.video_source = source

    @staticmethod
    def frames():
        print("Camera.video_source={}".format(Camera.video_source))
        camera = cv.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            rows = img.shape[0]
            cols = img.shape[1]

            cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
            cvOut = cvNet.forward()

            for detection in cvOut[0,0,:,:]:
                score = float(detection[2])
                if score > confThreshold:
                    left = detection[3] * cols
                    top = detection[4] * rows
                    right = detection[5] * cols
                    bottom = detection[6] * rows
                    cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

            # encode as a jpeg image and return it
            yield cv.imencode('.jpg', img)[1].tobytes()
