#!/bin/bash
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT

CAMERA=edgetpu_posenet
unset INDEX

CAMERA_PORT=5000
CAMERA_SOURCE=0

export FPS CAMERA INDEX
gunicorn3 --threads 5 --workers 1 --bind 0.0.0.0:$CAMERA_PORT app:app
