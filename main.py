#Pacotes
import cv2
import numpy as np
import argparse

#constantes
PROTOTXT="deploy.prototxt.txt"
CAFFEMODEL="res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD= 0.85

img=cv2.imread('o-FELICIDADE-facebook.jpg')
resized=cv2.resize(img, (300,300))
h,w=resized.shape[:2]
#cv2.imshow('image', resized)
#cv2.waitKey(0)
net= cv2.dnn.readNetFromCaffe(PROTOTXT,CAFFEMODEL)
blob=cv2.dnn.blobFromImage(resized,1,(300,300),(104,117,123))
net.setInput(blob)
predicts=net.forward()

# 200 previsões
for i in range(0, predicts.shape[2]):
    #probabilidade de cada previsão
    conf= predicts[0,0,i,2]
    if conf >= CONFIDENCE_THRESHOLD:
        box= predicts[0,0,i,3:7]*np.array([w,h,w,h])
        (startX, startY,endX, endY)=box.astype('int')
        cv2.rectangle(resized,(startX,startY),(endX,endY), (0,255,0), 2)
        #text = "{:.2f}%".format(conf*100)
        #cv2.putText(resized,text,(startY, startX),cv2.FONT_HERSHEY_SIMPLEX,0.25,(0,255,0),1)

cv2.imshow('image', resized)
cv2.waitKey(0)





