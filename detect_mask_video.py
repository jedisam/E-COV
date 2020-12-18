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

yid_counter = 0
kal_counter = 0

person_to_be_sent = ''
img1 = face_recognition.load_image_file('yid5.jpg')
img2 = face_recognition.load_image_file('kl.jpg')

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
    print(detections.shape)

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
        if person_to_be_sent != '' and person_to_be_sent == 'Yididya Samuel' and yid_counter == 0:
            print('The Person to be sent to: ', person_to_be_sent)
            # email sending URL
            URL = "https://facemask-alert.herokuapp.com/send?name=" + person_to_be_sent
            # URL = 'http://localhost:9000/send?name=' + person_to_be_sent

            # send request to email sending api
            r = requests.post(url=URL)
            print('The result is: ', r.json())
            yid_counter += 1
    return (locs, preds, person)


# load our serialized face detector model from disk
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 800 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds, person) = detect_and_predict_mask(frame, faceNet, maskNet)

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

        # include the probability in the label
        # label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
