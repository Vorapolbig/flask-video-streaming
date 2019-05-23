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

modelDir='../model/yolov2'
# load the COCO class labels
labelsPath = os.path.sep.join([modelDir, "coco.names"])
labels = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
classColors = np.random.randint(0, 255, size=(len(labels), 3),
                                dtype="uint8")

weightsPath = os.path.sep.join([modelDir, "yolov2.weights"])
configPath = os.path.sep.join([modelDir, "yolov2.cfg"])
if os.environ.get('THRESHOLD'):
    confThreshold = float(os.environ.get('THRESHOLD'))
else:
    confThreshold = 0.4
nmsThreshold = 0.3

cvNet = cv.dnn.readNetFromDarknet(configPath, weightsPath)
ln = cvNet.getLayerNames()
ln = [ln[i[0] - 1] for i in cvNet.getUnconnectedOutLayers()]

tickfreq = cv.getTickFrequency()
stop = False
frame = np.ndarray(shape=(480,640,3))
rectangles = []
texts = []
network_time = -1
person_cnt = 0

if os.environ.get('FPS') and os.environ.get('FPS') == "True":
    showfps = True
else:
    showfps = False

def get_boxes():
    global stop
    global network_time
    global rectangles
    global texts
    global person_cnt
    global frame

    while not stop:
        (H, W) = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)

        network_start = time.time()
        cvNet.setInput(blob)
        cvOut = cvNet.forward(ln)
        network_time = time.time() - network_start

        classIds = []
        confidences = []
        boxes = []

        rectangles = []
        texts = []

        # loop over each of the layer outputs
        for output in cvOut:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classId  = np.argmax(scores)
                confidence = scores[classId]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > confThreshold:
                    # scale the bounding box coordinates back relative to
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIds.append(classId)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

        person_cnt = 0
        # ensure at least one detection exists
        if len(indices) > 0:
            # loop over the indexes we are keeping
            for i in indices.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in classColors[classIds[i]]]
                rectangles.append( [ (x, y), (x + w, y + h), color, 2 ])
                if classIds[i] == 0:
                    person_cnt += 1
                    text = "{}: {:d}".format(labels[classIds[i]], person_cnt)
                else:
                    text = "{}: {:.4f}".format(labels[classIds[i]], confidences[i])
                texts.append([ text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2])


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
        global frame
        global rectangles
        global texts
        global network_time
        global person_cnt
        global showfps

        print("Camera.video_source={}".format(Camera.video_source))
        vs = WebcamVideoStream(src=Camera.video_source).start()
#       camera = cv.VideoCapture(Camera.video_source)
#       if not camera.isOpened():
#           raise RuntimeError('Could not start camera.')

        frame = vs.read()
        (H, W) = frame.shape[:2]
        print(H,W)
        numframes = 0
        person_cnt = 0

        # start the thread to read frames from the video stream
        t = Thread(target=get_boxes, name='get_boxes')
        t.daemon = True
        t.start()

        while True:
            # read current frame
            # ret, frame = camera.read()
            frame = vs.read()
            numframes += 1;

            for r in rectangles:
                p1, p2, p3, p4 = r
                cv.rectangle(frame, p1, p2, p3 , p4)
            for t in texts:
                p1, p2, p3, p4, p5, p6 = t
                cv.putText(frame, p1, p2, p3, p4, p5, p6)

            if showfps:
                now = cv.getTickCount()
                fps = tickfreq / (now - Camera.start)
                Camera.start = now
                title = "FPS: {:5,.0f}, Number of People: {:<3d}".format(round(fps,0), person_cnt)
            else:
                title = "Number of People: {:<3d}".format(person_cnt)

            cv.putText(frame, title, (20,40), cv.FONT_HERSHEY_COMPLEX, 1.0, (10,10,10), thickness=2)
            cv.putText(frame, title, (20,40), cv.FONT_HERSHEY_COMPLEX, 1.0, (250,250,250), thickness=1)
            # encode as a jpeg image and return it
            yield cv.imencode('.jpg', frame)[1].tobytes()
