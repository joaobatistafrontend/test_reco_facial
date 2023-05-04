import cv2
import mediapipe as mp
from deepface import DeepFace
import threading


webcam = cv2.VideoCapture(0)
recognition = mp.solutions.face_detection
recognize = recognition.FaceDetection()
desing = mp.solutions.drawing_utils
img = cv2.imread('4.jpg')

count = 0
cara = False


def comparar(frame,img):
    diff = cv2.absdiff(img,frame)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(diff_gray, 20, 255, cv2.THRESH_BINARY)
    return thresh

def verificar_face():
    global face
    try:
        if DeepFace.verify(frame,img.copy()['verified']):
            face = True
            print('vdd')

        else:
            face = False
            print('nao')

    except ValueError:
        face = False

while True:

    verificar, frame = webcam.read()

    if not verificar:
        break
    if verificar:
        if count % 30 == 0:
            try:
                threading.Thread(target=verificar_face,args=(frame.copy(),)).start()
            except ValueError:
                pass
        count += 1


    list_face = recognize.process(frame)

    if list_face.detections:
        for face in list_face.detections:
            desing.draw_detection(frame, face)



    if cv2.waitKey(5) == 27:
        break


difere = comparar(frame,img)
cv2.imshow('Difference', difere)
print(difere)

webcam.release()
cv2.destroyAllWindows()