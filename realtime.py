# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:21:57 2020

@author: BiO
"""


from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QInputDialog, QLineEdit, QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import cv2
from PIL import Image as img
import numpy as np 
import pandas as pd 
import tensorflow as tf
from scipy.stats import itemfreq


#Khai báo 
height = 30
width = 30
#model load
# load json and create model
json_file = open('E:/Python/trafficsignsCNN/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("E:/Python/trafficsignsCNN/model.h5")

result=pd.read_csv("E:/Python/trafficsignsCNN/NameTV.csv")


       
clicked = False      
 #Hàm nhận dạng 2
def Identification2():
    try:
        image=cv2.imread('E:/Python/trafficsignsCNN/camera.png')
        #path=self.labelPath.text() #Lấy đường dẫn của hình ảnh
        #image=cv2.imread(path) #Đọc hình ảnh
        data=[]
        #Xử lý hình ảnh
        image_from_array = img.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        data.append(np.array(size_image))
         
        X_test=np.array(data)
        X_test = X_test.astype('float32')/255 
        pred = model.predict_classes(X_test) #Nhận dạng
        
        
        for f in result['ClassId']: 
            if str(f) == str(pred[0]):
                id = int(f)
                print("Tên: "+result['Name'][id])
                global Name
                Name = "Name: "+result['Name'][id]
                break
    except:
        msg = QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText("Chưa chọn ảnh")    
        x=msg.exec_()
            

def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True        

url = "http://192.168.1.3:4747/video"
       
#-------------------------------------
global Name
try:
    cameraCapture = cv2.VideoCapture(url) 
except:
    cameraCapture = cv2.VideoCapture(0) 
cv2.namedWindow('camera')
cv2.setMouseCallback('camera', onMouse)

# Read and process frames in loop
success, frame = cameraCapture.read()  
clicked = False       
while success and not clicked:
    cv2.waitKey(1)
    success, frame = cameraCapture.read()    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 37)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                              1, 500, param1=50, param2=50)      
    if not circles is None:
        circles = np.uint16(np.around(circles))
        max_r, max_i = 0, 0
        for i in range(len(circles[:, :, 2][0])):
            if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                max_i = i
                max_r = circles[:, :, 2][0][i]               
        x, y, r = circles[:, :, :][0][max_i]            
        data=[]
        if y > r and x > r:
            image = frame[y-r-50:y+r+50, x-r-50:x+r+50]  
            #print(image)
        try:
            #time.sleep(0.2)   
            cv2.imwrite('E:/Python/trafficsignsCNN/camera.png', image) 
        except:
            print('wait')
         
        Identification2()
       
        for i in circles[0, :]:
            #cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            #cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            #Vẽ hình vuông xung quanh
            cv2.rectangle(frame, (i[0]+i[2], i[1]+i[2]), (i[0]-i[2], i[1]-i[2]), (255, 0, 0), 2)
            cv2.putText(frame, Name, (i[0]-i[2], i[1]-i[2]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA )
    
    cv2.imshow('camera', frame)   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cameraCapture.release()
        
        
        