import cv2
cam=cv2.VideoCapture(0)
cam.set(3,640)#width
cam.set(4,480)#height
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id=input("Enter a numeric user id: ")
print("taking samples look at camera")
count=0
while True:
    ret,img=cam.read()
    converted_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(converted_img,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count+=1
        #img is saved in jpg format in samples folder
        cv2.imwrite("samples/face."+str(face_id)+'.'+str(count)+".jpg",converted_img[y:y+h,x:x+w])
        cv2.imshow('image',img)
        
    k=cv2.waitKey(100)&0xff #waits for esc to be pressed key
    if(k==27):
        break
    elif count>=10:
        break
    
print("Smaples taken")
cam.release()
cv2.destroyAllWindows()