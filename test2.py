import cv2
from deepface import DeepFace
import mediapipe as mp
webcam = cv2.VideoCapture(0)
recognition = mp.solutions.face_detection
recognize = recognition.FaceDetection()
desing = mp.solutions.drawing_utils
img = cv2.imread('1.jpg')
facial = DeepFace.verify(img,webcam)
print(facial.items())
print(facial.get('verified'))
vdd = facial.se