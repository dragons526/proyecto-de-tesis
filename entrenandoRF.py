import cv2
import os
import numpy as np

dataPath = 'C:/Users/HP/Desktop/Reconocimiento Facial/Data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facedata = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + "/" + nameDir
    print("Leyendo las imagenes")

    for fileName in os.listdir(personPath):
        print("Rostros: ", nameDir + "/" + fileName)
        labels.append(label)
        facedata.append(cv2.imread(personPath+"/"+fileName))
        image = cv2.imread(personPath+"/"+fileName,0)
        #cv2.imshow('image',image)
        #cv2.waitkey(10)
    label = label + 1

#print('label= ',labels)
#print('Numero de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
#print('Numero de etiquetas 0: ',np.count_nonzero(np.array(labels)==1))

face_recognizer = cv2.face.EigenFaceRecognizer_create()


print("Entrenando...")
face_recognizer.train(facedata, np.array(labels))

#Almacenando el modelo obtenido
face_recognizer.write('modeloEigenFace.xml')



