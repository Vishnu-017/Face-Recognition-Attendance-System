import cv2
import numpy as np
from PIL import Image
import os
path ='samples'
recognizer=cv2.face.LBPHFaceRecognizer_create()#local binary patterns histogram
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def images_and_labels(path):
    imagepaths=[os.path.join(path,f) for f in os.listdir(path)]
    facesamples=[]
    ids=[]
    for imagepath in imagepaths:
        gray_img=Image.open(imagepath).convert('L')#convert to grayscale
        print(gray_img)
        img_arr=np.array(gray_img,'uint8')#unsign integer of 8 bit
        print(img_arr)
        print(os.path.split(imagepath)[-1])#face.1.8.jpg
        id=int(os.path.split(imagepath)[-1].split(".")[1])
        print(id)#1
        faces=detector.detectMultiScale(img_arr)
        for (x,y,w,h) in faces:
            facesamples.append(img_arr[y:y+h,x:x+w])
            ids.append(id)
    return facesamples,ids 

print("traing faces it will take few seconds") 
face,ids = images_and_labels(path)
print(face,"                ",ids)
recognizer.train(face,np.array(ids))
recognizer.write('trainer/trainer.yml') #the trained dataset is saved in yml format
print("model trained")


