import cv2
import os

dataPath = 'reconocimiento_facial/Data'
imagePaths = os.listdir(dataPath)
print('imagePaths: ', imagePaths)

face_recognizer = cv2.face.EigenFaceRecognizer_create()

face_recognizer.read('modeloEigenFace.xml')

cap = cv2.VideoCapture(0)

face_Classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    _, frame = cap.read()
    if _ == False:break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = face_Classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro, (150,150),interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        if result[1] < 5700:
            cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)

        if result[1] < 5700:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x,y-25), 1, 1.3, (255,0,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.putText(frame, '{}'.format('Desconocido'), (x, y - 25), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27: break

cap.release()
cv2.destroyAllWindows()