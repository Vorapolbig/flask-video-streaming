#!/usr/bin/env python
# Copyright (c) 2014 Miguel Grinberg
# Copyright 2019 NXP

# SPDX-License-Identifier: MIT

from importlib import import_module
import os
from flask import Flask, render_template, Response

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

#   Use camera based index if it exits.
    index = 'index' + camera_suffix + '.html'
    if not os.path.exists(os.path.sep.join(['templates', index])):
        index = 'index.html'

    return render_template(index)


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
