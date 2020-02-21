#!/bin/bash

# Copyright 2020 NXP
# SPDX-License-Identifier: MIT

CAMERA=edgetpu_detection
INDEX=index_coco_tpu.html

CAMERA_PORT=5000
CAMERA_SOURCE=0

FPS=True

export FPS CAMERA INDEX
gunicorn3 --threads 5 --workers 1 --bind 0.0.0.0:$CAMERA_PORT app:app
