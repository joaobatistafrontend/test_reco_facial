import cv2
import os
from deepface import DeepFace
import mediapipe as mp


def FacialRecognition():
    webcam = cv2.VideoCapture(0)
    recognition = mp.solutions.face_detection
    recognize = recognition.FaceDetection()
    desing = mp.solutions.drawing_utils

    while True:
        checker, frame = webcam.read()
        if not checker:
            break

        list_face = recognize.process(frame)
        if list_face.detections:
            for face in list_face.detections:
                desing.draw_detection(frame, face)




        cv2.imshow('facial recognition', frame)

        if cv2.waitKey(5) == 27:
            break




    webcam.release()


FacialRecognition()

