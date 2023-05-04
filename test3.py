import cv2
import mediapipe as mp
from deepface import DeepFace


def FacialRecognition():
    webcam = cv2.VideoCapture(0)
    recognition = mp.solutions.face_detection
    recognize = recognition.FaceDetection()
    desing = mp.solutions.drawing_utils

    img = cv2.imread('4.jpg')



    while True:
        checker, frame = webcam.read()
        if not checker:
            break

        list_face = recognize.process(frame)

        if list_face.detections:
            for face in list_face.detections:
                desing.draw_detection(frame, face)
                facial = DeepFace.verify(img,frame)
                if face == facial:
                    print('pretou')

        cv2.imshow('facial recognition', frame)

        if cv2.waitKey(5) == 27:
            break

    webcam.release()


FacialRecognition()