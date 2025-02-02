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
# Print in the following print line will show you 
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

'''IDENTIFY FROM IMAGE'''

# Change the name of the following variable (img) 
# based on the name of the image file that you want to use
img = cv2.imread("boy.jpg") # boy.jpg and stop_sign.jpg are already given
plt.imshow(img)

ClassIndex, confidence, bbox = model.detect(img, confThreshold = 0.5)
print(ClassIndex)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (225, 0, 0), 2)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color = (0, 255, 0), thickness = 3)
    
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()