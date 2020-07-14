#!/usr/bin/env python
# Copyright (c) 2014 Miguel Grinberg
# Copyright 2019, 2020 NXP

# SPDX-License-Identifier: MIT

from importlib import import_module
import os
from flask import Flask, render_template, Response

import cv2 
import time 

# import camera driver
if os.environ.get('CAMERA'):
    camera_suffix = '_' + os.environ['CAMERA']
    Camera = import_module('camera' + camera_suffix).Camera
else:
    camera_suffix = ''
    from camera import Camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

if os.environ.get('CAMERA_SOURCE'):
    Camera.set_video_source(os.environ['CAMERA_SOURCE'])

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""

    global camera_suffix
    index = 'index.html'

    if os.environ.get('INDEX') and os.path.exists(os.path.sep.join(['templates', os.environ.get('INDEX')])):
        index = os.environ.get('INDEX')
    elif os.path.exists(os.path.sep.join(['templates', 'index' + camera_suffix + '.html'])):
        #   Use camera based index if it exits.
        index = 'index' + camera_suffix + '.html'

    return render_template(index)


def gen_cam(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
def gen():
    
    cap = cv2.VideoCapture('rtsp://service:!Service0@192.168.4.82:554/h264')
    
    while True:
        ret, img = cap.read()
        if  ret == True:
            frame = cv2. imencode('.jpg',img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else:
            break
            
    
    

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)

