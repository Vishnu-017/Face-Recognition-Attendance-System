import cv2
from datetime import datetime
recognizer=cv2.face.LBPHFaceRecognizer_create()#local binary patterns histograms
recognizer.read('trainer/trainer.yml')
casecadepath="haarcascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(casecadepath)

font=cv2.FONT_HERSHEY_SIMPLEX
id=2
names=['','vishnu'] #frist empty bcoz starts with 0

cam =cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(3,640)#width
cam.set(4,480)#height
#defining min window size to be recognized
minW =0.1*cam.get(3)
minH=0.1*cam.get(4)

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
            

while True:
    ret,img=cam.read()
    converted_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(
        converted_img,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW),int(minH)),
        )
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,accuracy=recognizer.predict(converted_img[y:y+h,x:x+h])
        print(id,"          ",accuracy)#used to predict
        
        idlist=[]
        if(accuracy<100):
            
            name=names[id]#name of the person with id
            idlist.append(name)
            accuracy=" {0}%".format(round(100-accuracy))
            attendance(name)
            
                
        else:
            id="unknown"
            accuracy=" {0}%".format(round(100-accuracy))
        #for drawing rectangle and text
        cv2.putText(img,str(name),(x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(img,str(accuracy),(x+25,y-25),font,1,(255,255,0),1)
        cv2.waitKey(100)
    '''if name in idlist:
        print(name)
        break'''
    cv2.imshow('camera',img)
    k=cv2.waitKey(100)&0xff #press esc for exit
    if k==27:
        break
    
print("thanks for using")
cam.release()
cv2.destroyAllWindows()
    
        
            