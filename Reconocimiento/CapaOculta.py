from cProfile import label
import imp
import cv2
import os
import numpy as np
from time import time     

dataRuta='C:/Users/vicen/Desktop/Reconocimiento Facial Py cv/ReconocimientoFacialPyCv/Data'
listaData=os.listdir(dataRuta)

ids=[]
rostrosData=[]
id=0
tiempoInicial = time()

for fila in listaData:  
    rutaCompleta=dataRuta+'/'+fila
    print('Iniciando Lectura...')
    for archivo in os.listdir(rutaCompleta):
        print('Imagenes: ',fila+'/'+archivo)
        ids.append(id)
        rostrosData.append(cv2.imread(rutaCompleta+'/'+archivo,0))
    id=id+1
    tiempoFinalLectura=time()
    tiempoTotalLectura=tiempoFinalLectura-tiempoInicial
    print('Tiempo Total Lectura: ',tiempoTotalLectura)

entrenamientoModelo1=cv2.face.EigenFaceRecognizer_create()
print('Iniciando el entrenamiento... Espere')
entrenamientoModelo1.train(rostrosData,np.array(ids))

tiempoFinalEntrenamiento=time()
tiempoTotalEntrenamiento=tiempoFinalEntrenamiento-tiempoTotalLectura
print('Tiempo entrenamiento Total: ',tiempoFinalEntrenamiento)
entrenamientoModelo1.write('EntrenamientoEigenfacerecognizer.xml')
print('Entrenamiento Concluido')