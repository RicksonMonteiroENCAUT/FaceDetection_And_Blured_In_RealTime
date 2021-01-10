import cv2
import numpy as np
from time import sleep
'''
Elementos da predição
conf = inference_results[0, 0, i, 2] # extrair a confiança (ou seja, probabilidade)

idx = int(inference_results[0, 0, i,1]) # extrair o índice do rótulo da classe

boxPoints = inference_results[0, 0, i, 3:7] 
'''

#constantes
PROTOTXT="deploy.prototxt.txt"
MODEL="res10_300x300_ssd_iter_140000.caffemodel"
net=cv2.dnn.readNetFromCaffe(PROTOTXT,MODEL)
CONFIDENCE_THRESHOLD=0.85
def anonymize_face_pixelate(image, blocks=9):
    """
    Função extraída do autor Adrian Rosebrock do site PyImageSearch
    https://www.pyimagesearch.com
    """

    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (B, G, R), -1)
    # return the pixelated blurred image
    return image


def preprocessing(frame):
    h,w=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame)
    net.setInput(blob)
    predicts= net.forward()
    for i in range(0, predicts.shape[2]):
        conf=predicts[0,0,i,2]
        if conf>= CONFIDENCE_THRESHOLD:
            (startX, startY, endX, endY)=(predicts[0,0,i,3:7]*np.array([w,h,w,h])).astype('int')
            #cv2.rectangle(frame,(startX, startY), (endX, endY),(0,255,0),2)
            face= frame[startY:endY,startX:endX]
            kW=int(w/3.0)
            kH=int(h/3.0)
            #blured= cv2.GaussianBlur(face,(kW,kH),0)
            blured = anonymize_face_pixelate(face)
            frame[startY:endY, startX:endX]=blured

#iniciando a câmera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
sleep(2.0)

while cap.isOpened():
    #capturando frame
    _,frame= cap.read()
    frame=cv2.resize(frame,(400,400))
    preprocessing(frame)
    cv2.imshow('image', frame)
    key=cv2.waitKey(1) & 0xff
    if key== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


