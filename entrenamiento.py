import cv2
import os
import numpy as np

dataPath = 'reconocimiento_facial/Data'
peopleList = os.listdir(dataPath)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('leyendo las imagenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
        #cv2.imshow('image', image)
        #cv2.waitKey(10)
    label = label + 1

#print('labels= ', labels)

face_recognizer = cv2.face.EigenFaceRecognizer_create()

print('entrenando...')
face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('modeloEigenFace.xml')
print("modelo almacenado...")