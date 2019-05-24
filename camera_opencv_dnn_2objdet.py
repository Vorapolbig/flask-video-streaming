# Copyright 2019 NXP
# SPDX-License-Identifier: MIT

# Based on example code from:
#  opencv/samples/dnn/object_detection.py,
#  flask-video-streaming/camera_opencv.py, and
#  imutils/demos/fps_demo.py

import cv2 as cv
from base_camera import BaseCamera

from imutils.video import WebcamVideoStream
import numpy as np
import time
import os
from threading import Thread

modelDir='../model/2object_detect'
labelsPath = os.path.sep.join([modelDir, "3object_detect_classes.txt"])
labels = open(labelsPath).read().strip().split("\n")

# BGR
classColors = [ (23, 230, 210), (0, 165, 255), (192, 192, 192)]
if len(labels) > 3:
    classColors = np.append(classColors, np.random.randint(0, 255, size=(len(labels)-3, 3), dtype="uint8", axis=0))

model = os.path.sep.join([modelDir, "frozen_inference_graph.pb"])
graph = os.path.sep.join([modelDir, "graph.pbtxt"])
if os.environ.get('THRESHOLD'):
    confThreshold = float(os.environ.get('THRESHOLD'))
else:
    confThreshold = 0.6
nmsThreshold = 0.4

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
                    left = int(detection[3] * cols)
                    top = int(detection[4] * rows)
                    right = int(detection[5] * cols)
                    bottom = int(detection[6] * rows)
                    id = int(detection[1] - 1)
#                    print(score, id, left, top, right, bottom)
                    color = classColors[id]
                    cv.rectangle(img, (left, top), (right, bottom), color, thickness=2)
                    text = "{}: {:.2f}".format(labels[id], score)
                    cv.putText(img, text, (left, top - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # encode as a jpeg image and return it
            yield cv.imencode('.jpg', img)[1].tobytes()
