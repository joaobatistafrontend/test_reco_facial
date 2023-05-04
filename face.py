import cv2
import threading
from deepface import DeepFace

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

count = 0
face = False
img = cv2.imread('5.jpg')

def verificar_face():
    global face
    try:
        if DeepFace.verify(frame,img.copy()['verified']):
            face = True
        else:
            face = False
    except ValueError:
        face = False

while True:
    rest, frame = cam.read()

    if rest:
        if count % 30 == 0:
            try:
                threading.Thread(target=verificar_face,args=(frame.copy(),)).start()
            except ValueError:
                pass
        count += 1

        if face:
            cv2.putText(frame,"Altenticado",(20,450), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0), 3)
        else:
            cv2.putText(frame,"Não Altenticado",(20,450), cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255), 3)

        cv2.imshow('Camera',frame)


    if cv2.waitKey(5) == 27:
        break

cv2.destroyAllWindows()