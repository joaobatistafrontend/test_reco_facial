
import cv2
import os
from deepface import DeepFace
import mediapipe as mp


def FacialRecognition():
     webcam = cv2.VideoCapture(0)
     reconhecer_face = mp.solutions.face_detection
     reconhecer = reconhecer_face.FaceDetection()
     desenho = mp.solutions.drawing_utils
     imh = cv2.imread('5.jpg')

     while True:
          checker, frame = webcam.read()
          if not checker:
               break

          list_face = reconhecer.process(frame)
          if list_face.detections:
               for face in list_face.detections:
                    desenho.draw_detection(frame, face)
          cv2.imshow('facial recognition', frame)

          if cv2.waitKey(5) == 27:
               break

     webcam.release()
     cv2.destroyWindow()


FacialRecognition()

