from asyncore import read
import cv2
import os
import imutils

dataRuta = 'C:/Users/vicen/Desktop/Reconocimiento Facial Py/ReconocimientoFacialPy/Data'
listaData = os.listdir(dataRuta)
entrenamientoEigenFaceRecognizer=cv2.face.EigenFaceRecognizer_create()
entrenamientoEigenFaceRecognizer.read('EntrenamientoEigenfacerecognizer.xml')

ruidos = cv2.CascadeClassifier('C:\\Users\\vicen\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
CapturaVideo=cv2.VideoCapture(0)
while True:
    respuesta,captura = CapturaVideo.read()
    if respuesta==False:
        break
    captura = imutils.resize(captura,width=640)
    grises = cv2.cvtColor(captura, cv2.COLOR_BGR2GRAY)
    idCaptura=grises.copy()
    cara = ruidos.detectMultiScale(grises,1.3,5)

    for(x,y,e1,e2) in cara:
        rostroCapturado = idCaptura[y:y+e2,x:x+e1]
        rostroCapturado = cv2.resize(rostroCapturado, (160,160), interpolation = cv2.INTER_CUBIC)
        resultado = entrenamientoEigenFaceRecognizer.predict(rostroCapturado)
        cv2.putText(captura, '{}'.format(resultado),(x,y-5),1,1.3,(255,0,0),1,cv2.LINE_AA)

        if resultado[1]<9000:
            cv2.putText(captura, '{}'.format(listaData[resultado[0]]),(x,y-20),2,1,(255,0,0),1,cv2.LINE_AA)
            cv2.rectangle(captura,(x,y),(x+e1,y+e2),(255,0,0),2)
        else:
            cv2.putText(captura,'no encontrado',(x,y-20),2,1.1,(255,0,0),1,cv2.LINE_AA)
            cv2.rectangle(captura,(x,y),(x+e1,y+e2),(255,0,0),2)

    cv2.imshow("Resultado",captura)
    if(cv2.waitKey(1)==ord('q')):
        break
CapturaVideo.release()
cv2.destroyAllWindows()

