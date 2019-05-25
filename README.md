NXP-opencv_dnn flask-video-streaming 
====================================

Flask video streaming with video processing using OpenCV's DNN support.

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


OpenCV scripts created by Edward Swarthout.

Based on code from Miguel Grinberg's article [video streaming with Flask](http://blog.miguelgrinberg.com/post/video-streaming-with-flask) and its follow-up [Flask Video Streaming Revisited](http://blog.miguelgrinberg.com/post/flask-video-streaming-revisited).
