import cv2
import numpy as np
import face_recognition
import os 
from datetime import datetime
path = 'imgsample'
images = []
classnames = []
mylist = os.listdir(path)
print(mylist)
for cls in mylist:
    currimg=cv2.imread("{}/{}".format(path,cls))
    images.append(currimg)
    classnames.append(os.path.splitext(cls)[0]) #to remove .jpeg in img txt
print(classnames)

def findencodings(images):
    encodelist=[]
    for img in images:
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def attendance(name):
    with open('attendance.csv','r+') as f:
        namelist=[]
        mydatalist = f.readlines()
        for line in mydatalist:
            entry = line.split(',')
        if name not in namelist:
            now = datetime.now()
            datestring = now.strftime("%H:%M:%S")
            f.writelines(f'\n {name},{datestring}')


encodelistknown = findencodings(images)
print("Encoded done")

cap = cv2.VideoCapture(0) 

while True:
    succ,img=cap.read()
    imgsmall = cv2.resize(img,(0,0),None,0.25,0.25)  # toresize from original img
    imgsmall = cv2.cvtColor(imgsmall,cv2.COLOR_BGR2RGB)
    facesincurrframe = face_recognition.face_locations(imgsmall)#to find the locstions of img
    encodecurrframe = face_recognition.face_encodings(imgsmall,facesincurrframe) 

    for encodeface,faceloc in zip(encodecurrframe,facesincurrframe):
        matches = face_recognition.compare_faces(encodelistknown,encodeface)
        facedis = face_recognition.face_distance(encodelistknown,encodeface)
        #lowest dis is best match 
        #same face min dis 
        matchindex= np.argmin(facedis)
        if matches[matchindex]:
            name = classnames[matchindex].upper()
            #print(name) 
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 #to resize the recognising area and cover the face
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),thickness=2)
            attendance(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)

