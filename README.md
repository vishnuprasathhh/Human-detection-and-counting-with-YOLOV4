# Human-detection-and-counting-with-YOLOV4
Human Detection and Counting  This Python script is designed for pedestrian detection using a pre-trained YOLOv4-tiny model. The script utilizes the OpenCV library for computer vision tasks and provides options for detecting pedestrians in images, videos, or live we) object detection framework to detect pedestrians.

Usage To use the script, you can run it from the command line with the following arguments:

--input_type: Specify the type of input data (choices: image, video, webcam). --input_path: Specify the path to the input file (for images or videos) or use the default webcam feed (for webcam). To detect pedestrians in an image: python script.py --input_type image --input_path path/to/image.jpg To detect pedestrians in a video: python script.py --input_type video --input_path path/to/video.mp4 To detect pedestrians in a webcam: python script.py --input_type webcam

Ensure that you have the necessary model files (yolov4-tiny.weights, yolov4-tiny.cfg) and label file (coco.names) in the same directory as the script.

Requirements Python 3.x OpenCV (opencv-python package) NumPy imutils argparse

