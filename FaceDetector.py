import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('Detect/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("Recognizer\\trainingData.xml")
ID = 0
fontface=cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontColor = (0,255,0)

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ID, conf=rec.predict(gray[y:y+h, x:x+w])
        if(ID==1):
            ID="Souliya"
        elif(ID==2):
            ID="Athid"
        elif(ID==3):
            ID="Phimon"
        cv2.putText(img, str(ID), (x, y+h), fontface, fontScale, fontColor )
    cv2.imshow("Face", img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()