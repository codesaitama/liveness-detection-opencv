# import the necessary packages
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

protoPath = os.path.sep.join(['face_detector', "deploy.prototxt"])
modelPath = os.path.sep.join(['face_detector', "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model('model')
le = pickle.loads(open('le.pickle', "rb").read())

# grab the frame from the image and resize it
# to have a maximum width of 600 pixels
frame = cv2.imread(args["input"])  #vs.read()
#print(frame)

frame = imutils.resize(frame, width=600)

# grab the frame dimensions and convert it to a blob
(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and
# predictions
net.setInput(blob)
detections = net.forward()

result = {}

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]
    # filter out weak detections
    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for
        # the face and extract the face ROI
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # ensure the detected bounding box does fall outside the
        # dimensions of the frame
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)

        # extract the face ROI and then preproces it in the exact
        # same manner as our training data
        face = frame[startY:endY, startX:endX]
        face = cv2.resize(face, (32, 32))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        # pass the face ROI through the trained liveness detector
        # model to determine if the face is "real" or "fake"
        preds = model.predict(face)[0]
        j = np.argmax(preds)
        label = le.classes_[j]

        #print(le.classes_[j])
        print('-----' * 20)
        print(label);
        print(preds[j])
        print('-----' * 20)