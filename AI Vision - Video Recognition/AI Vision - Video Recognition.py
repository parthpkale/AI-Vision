import cv2
from matplotlib import pyplot as plt

# COCO is a large image database which has 
# 1000s of images which my computer vision can be trained on.
config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
# frozen_inference_graph is a pre-trained deep learning model
# frozen means it is pre-trained and can not be changed
frozen_model = "frozen_inference_graph.pb"

# The Deep Neural Network (DNN) module is used to 
# load a pre-trained object detection model
# The following command loads the model onto openCV
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# The following 4 lines put the contents of the
# labels file into the array classLabels
classLabels = []
file_name = "labels.txt"
with open(file_name, "rt") as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
# Printin the following print line will show you 
# all the objects that COCO can detect
#print(classLabels)    
#print(len(classLabels))

# Size of frame
model.setInputSize(320, 320)
# Scale factor of frame
model.setInputScale(1.0/127.5)
# Mean value for frame
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

'''IDENTIFY FROM VIDEO'''

import numpy as np

# Change the name of the following variable (img) 
# based on the name of the video file that you want to use
cap = cv2.VideoCapture("london_street.mp4") # london_street.mp4 is already given
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Can't open the video.")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Exiting.")
        break

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    print(ClassIndex)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (225, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0] + 10, boxes[1] + 40),
                            font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(2) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()