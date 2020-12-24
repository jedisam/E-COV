# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import face_recognition
import requests
import json
from os.path import dirname, join

yid_counter = 0
kal_counter = 0

person_to_be_sent = ''
img1 = face_recognition.load_image_file('./static/assets/img/avatars/yid.jpg')
img2 = face_recognition.load_image_file('./static/assets/img/avatars/kal.jpg')

face_encoding1 = face_recognition.face_encodings(img1)[0]
face_encoding2 = face_recognition.face_encodings(img2)[0]

known_face_encoding = [
    face_encoding1,
    face_encoding2,
]


def recognize_face(frame):
    name = "unknown"
    face_encoding_unknown = face_recognition.face_encodings(frame)

    for unknown_face_encoding in face_encoding_unknown:
        res = face_recognition.compare_faces(
            known_face_encoding, unknown_face_encoding)
        print("res: ", res)

        if res[0]:
            name = "Yididya Samuel"
        elif res[1]:
            name = "Kalkidan Samuel"
        # elif res[2]:
        #     name = "KAL2nd"
        print('NAME IS: ', name)

        print(f'Found {name} in the picture!')

        return name


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    withoutMask = 0
    withMask = 0
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        print(preds)
        withoutMask = preds[0][1]
        withMask = preds[0][0]
        print("WITHOUT MASK") if withoutMask > withMask else print(
            "=======================WITH MASK===================================")

    person = ''

    if withoutMask > withMask:
        person = recognize_face(frame)
        print('Here is the name: ', person)
        person_to_be_sent = person
        global yid_counter
        global kal_counter
        if person_to_be_sent != '' and person_to_be_sent == 'Yididya Samuel' and yid_counter == 0:
            print('The Person to be sent to: ', person_to_be_sent)
            # email sending URL
            URL = "https://facemask-alert.herokuapp.com/send?name=" + person_to_be_sent
            # URL = 'http://localhost:9000/send?name=' + person_to_be_sent
            print('Yeah')
            # send request to email sending api
            r = requests.post(url=URL)
            print('The result is: ', r.json())
            yid_counter += 1
        elif person_to_be_sent != '' and person_to_be_sent == 'Kalkidan Samuel' and kal_counter == 0:
            print('The Person to be sent to: ', person_to_be_sent)
            # email sending URL
            URL = "https://facemask-alert.herokuapp.com/send?name=" + person_to_be_sent
            # URL = 'http://localhost:9000/send?name=' + person_to_be_sent

            # send request to email sending api
            r = requests.post(url=URL)
            print('The result is: ', r.json())
            kal_counter += 1
    return (locs, preds, person)


# load our serialized face detector model from disk
prototxtPath = join(dirname(__file__), 'face_detector', "deploy.prototxt")
# weightsPath = join(dirname(__file__), 'face_detector',
#    "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
# prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = join(dirname(__file__), 'face_detector',
                   "res10_300x300_ssd_iter_140000.caffemodel")
# weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model(
    join(dirname(__file__), "face_mask.model"))
# maskNet = load_model("face_mask.model")


class VideoCamera(object):
    def __init__(self):
        self.stream = VideoStream(src=0).start()

    def __del__(self):
        self.stream.stop()

    def get_frame(self):
        # global image
        image = self.stream.read()
        image = cv2.flip(image, 1, 1)

        # predict and detect face mask
        (locs, preds, person) = detect_and_predict_mask(image, faceNet, maskNet)
        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            if person != '' or person != None:
                label = "Mask" if mask > withoutMask else f'Without_Mask {person}'
            else:
                label = "Mask" if mask > withoutMask else "No Mask"

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            pos = (startX + 15, startY -
                   10) if label == "Mask" else (startX - 55, startY - 10)
            # include the probability in the label
            # label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # image
            # font
            font = cv2.FONT_HERSHEY_TRIPLEX
            # fontScale
            fontScale = 0.7
            # White color in BGR
            color2 = (255, 255, 255)
            # Line thickness of 2 px
            thickness = 1
            cv2.putText(image, label, pos, font, fontScale, color2, thickness,
                        cv2.LINE_AA)
            # print('DISTANCE: ', endX - startX)
            # print('STARTX ', startX)
            # print('ENDX: ', endX)
            cv2.line(image, (startX, startY),
                     (startX + int((endX - startX) / 4), startY), color, 4)
            cv2.line(image, (int(startX + endX / 4), startY),
                     (endX, startY), color, 4)
            cv2.line(image, (startX, startY),
                     (startX, startY + int((endY - startY) / 4)), color, 4)
            cv2.line(image, (startX, endY - int((endY - startY)/4)),
                     (startX, endY),  color, 4)
            cv2.line(image, (startX, endY),
                     (startX + int((endX - startX) / 4), endY), color, 4)
            cv2.line(image, (int(startX + endX / 4), endY),
                     (endX, endY), color, 4)
            cv2.line(image, (endX, startY),
                     (endX, startY + int((endY - startY) / 4)), color, 4)
            cv2.line(image, (endX, endY - int((endY - startY)/4)),
                     (endX, endY),  color, 4)

        ret, jpeg = cv2.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())
        return data
