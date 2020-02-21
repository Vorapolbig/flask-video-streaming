NXP-opencv_dnn flask-video-streaming
====================================

Flask video streaming with video processing in various frameworks.

Coral Edge TPU Accelerator with Edge TPU API
----------------------------------------------------

1. camera_edgetpu_detection.py: Adapts [coral.ai/examples/detect-image](https://coral.ai/examples/detect-image/)

Uses the edgetpu.detect.engine and supports MobileNet SSD Object Detection models shown here [coral.ai/models](https://coral.ai/examples/detect-image/)

The default model and label file is

* ../model/coral_models/[mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite](https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite)
* ../model/coral_models/[coco_labels.txt](https://github.com/google-coral/edgetpu/raw/master/test_data/coco_labels.txt)

The non-edgetpu tflite model is also supported and will run on the CPU, but currently requires a Edge TPU to still be available.

Custom modules and labels are supported such as:

* ../model/coral_models/mobilenet_ssd_v2_goggle_quant_postprocess_edgetpu.tflite
* ../model/goggle_labels.txt

2. camera_edgetpu_posenet.py: Adapts [google-coral/project-posenet](https://github.com/google-coral/project-posenet)

Uses PoseEngine from pose_engine.py

The default model is

* ../model/posenet/[posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite](https://github.com/google-coral/project-posenet/blob/master/models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite)


OpenCV DNN support
-------------------

See wiki [Deep-Learning-in-OpenCV](https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV)

1. camera_opencv_dnn_wiki.py:  Adapts [TensorFlow-Object-Detection-API](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API)

To use, set CAMERA=opencv_dnn_wiki and place the tensorflow model in ../model.

Requires ../model/frozen_inference_graph.pb and ../model/graph.pbtxt

2. camera_opencv_yolov2.py: Darknet based YOLOv2 object detection and person counting.

Requires model files in ../model/yolov2/

* [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
* [yolov2.weights](https://pjreddie.com/media/files/yolov2.weights)
* [yolov2.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg)

For details see https://pjreddie.com/darknet/yolov2/

3. camera_opecv_2objdet.py: Tensorflow SSD with Mobilenet v1 trained on goggles and helmet.

Requires model files in ../model/2object_detect/

* graph.pbtxt
* frozen_inference_graph.pb
* 3object_detect_classes.txt

---

Scripts created by Edward Swarthout.

Based on code from Miguel Grinberg's article [video streaming with Flask](http://blog.miguelgrinberg.com/post/video-streaming-with-flask) and its follow-up [Flask Video Streaming Revisited](http://blog.miguelgrinberg.com/post/flask-video-streaming-revisited).
