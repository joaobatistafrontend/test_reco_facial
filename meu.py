import cv2
import mediapipe as mp
from deepface import DeepFace


cam = cv2.VideoCapture(0)
solucao_reconhecimento = mp.solutions.face_detection
reconhecedor_rosto = solucao_reconhecimento.FaceDetection()
desenho = mp.solutions.drawing_utils

img = ('01.jpg')
face = False
while True:
    verificador, frame = cam.read()
    if not verificador:
        break
    list_rosto = reconhecedor_rosto.process(frame)

    if list_rosto.detections:
        for rosto in list_rosto.detections:
            desenho.draw_detection(frame,rosto)
        if DeepFace.verify(frame,img).get('verified'):
            face = True
            cv2.putText(frame, "Altenticado", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            face = False
            cv2.putText(frame,"Não Altenticado",(20,450), cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255), 3)

    cv2.imshow('video',frame)

    if cv2.waitKey(5) == 27:
        break

cam.release()
cv2.destroyAllWindows()