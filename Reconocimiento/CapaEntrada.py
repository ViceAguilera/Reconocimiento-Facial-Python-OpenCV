import cv2
import os
import imutils

modelo = 'FotosVice'
ruta1 = 'C:/Users/vicen/Desktop/Reconocimiento Facial Py cv/ReconocimientoFacialPyCv'
rutacompleta = ruta1 + '/' + modelo
if  not os.path.exists(rutacompleta):
    os.makedirs(rutacompleta)

CapturaVideo = cv2.VideoCapture(0)
ruidos = cv2.CascadeClassifier('C:\\Users\\vicen\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
id = 350

while True:
    respuesta,captura = CapturaVideo.read()
    if respuesta == False:
        break
    captura = imutils.resize(captura,width=640)
    grises = cv2.cvtColor(captura, cv2.COLOR_BGR2GRAY)
    idCaptura = captura.copy()

    cara = ruidos.detectMultiScale(grises,1.4,5)

    for(x,y,e1,e2) in cara:
        cv2.rectangle(captura,(x,y),(x+e1,y+e2),(255,0,0),2)
        rostroCapturado = idCaptura[y:y+e2,x:x+e1]
        rostroCapturado = cv2.resize(rostroCapturado, (160,160), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id), rostroCapturado)
        id=id+1

    cv2.imshow("Resultado", captura)

    if id==500:
        break

CapturaVideo.release()
cv2.destroyAllWindows()