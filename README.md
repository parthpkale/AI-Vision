# AI-Vision

## Overview
This project leverages OpenCV and a pre-trained deep learning model to recognize and classify objects in images, videos, and live webcam feeds. The model used in this project is based on **SSD MobileNet V3**, which is trained on the **COCO dataset**—a large-scale dataset with 80 common object classes. The project allows for object detection and displays the detected objects with bounding boxes and labels.

## Features
- **Image-based Object Detection:** Detect objects in static images.
- **Video-based Object Detection:** Detect objects in video files.
- **Live Webcam Object Detection:** Detect objects using a connected webcam.
- **Real-time Object Classification:** Using a pre-trained model to classify detected objects in real time.

## Prerequisites
Before running the project, make sure you have the following dependencies installed:

- Python 3.x
- OpenCV (for image and video processing)
- Matplotlib (for image visualization)
- A pre-trained model file (`frozen_inference_graph.pb` and `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`) -> Both are provided above.

Install the required packages using `pip`:

```bash
pip install opencv-python opencv-python-headless matplotlib numpy
```

## Files & Configuration
- (`frozen_inference_graph.pb`): The pre-trained model used for object detection.
- (`ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`): The configuration file for the model.
- (`labels.txt`): A text file containing the class labels (objects) that the model can detect, corresponding to the COCO dataset.
- Test images/videos: Example files like boy.jpg, stop_sign.jpg, london_street.mp4, or webcam input can be used for object detection.
- Choose the python file that you wish to run based on your input (i.e. picture, video or camera).

## Enhanced Configuration
- **Model Configuration:** The model uses the **MobileNet V3 SSD** architecture, trained on the **COCO dataset**. You can modify the (`confThreshold`) to adjust the confidence level for object detection.
- **Window Size:** Adjust the window size where the video feed or image will be displayed by using (`cv2.resizeWindow()`).

## Conclusion
This project demonstrates how artificial intelligence, specifically deep learning and computer vision, can be applied to detect and classify objects in real-time from images, videos, and webcam streams. You can customize the code further to add more advanced features like tracking, multi-object recognition, or integration with other AI systems.


