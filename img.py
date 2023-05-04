import cv2
from deepface import DeepFace
import os

arqui = os.listdir()
for arquivo in arqui:
    if 'jpg' in arquivo:
        img = cv2.imread(arquivo)

        result = DeepFace.analyze(img, actions=('age', 'gender', 'race', 'emotion'),enforce_detection=False)

        print(result)