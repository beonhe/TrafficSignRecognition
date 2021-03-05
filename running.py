
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
import time


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

def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]

Name = '???' 
clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(991, 590)
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.labelNameProject = QtWidgets.QLabel(self.centralwidget)
        self.labelNameProject.setGeometry(QtCore.QRect(200, 0, 671, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(26)
        font.setBold(True)
        font.setWeight(75)
        self.labelNameProject.setFont(font)
        self.labelNameProject.setObjectName("labelNameProject")
        self.labelPath = QtWidgets.QLabel(self.centralwidget)
        self.labelPath.setGeometry(QtCore.QRect(40, 90, 411, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.labelPath.setFont(font)
        self.labelPath.setObjectName("labelPath")
        self.labelName = QtWidgets.QLabel(self.centralwidget)
        self.labelName.setGeometry(QtCore.QRect(520, 90, 551, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.labelName.setFont(font)
        self.labelName.setObjectName("labelName")
        self.labelIP = QtWidgets.QLabel(self.centralwidget)
        self.labelIP.setGeometry(QtCore.QRect(40, 130, 381, 331))
        self.labelIP.setText("")
        self.labelIP.setPixmap(QtGui.QPixmap("E:/Python/trafficsignsCNN/input.png"))
        self.labelIP.setScaledContents(True)
        self.labelIP.setObjectName("labelIP")
        self.labelOP = QtWidgets.QLabel(self.centralwidget)
        self.labelOP.setGeometry(QtCore.QRect(570, 130, 381, 331))
        self.labelOP.setText("")
        self.labelOP.setPixmap(QtGui.QPixmap("E:/Python/trafficsignsCNN/output.png"))
        self.labelOP.setScaledContents(True)
        self.labelOP.setObjectName("labelOP")
        self.btnIP = QtWidgets.QPushButton(self.centralwidget)
        self.btnIP.setGeometry(QtCore.QRect(150, 490, 141, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnIP.setFont(font)
        self.btnIP.setObjectName("btnIP")
        self.btnOP = QtWidgets.QPushButton(self.centralwidget)
        self.btnOP.setGeometry(QtCore.QRect(700, 490, 141, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnOP.setFont(font)
        self.btnOP.setObjectName("btnOP")
        self.btnOnCam = QtWidgets.QPushButton(self.centralwidget)
        self.btnOnCam.setGeometry(QtCore.QRect(420, 490, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btnOnCam.setFont(font)
        self.btnOnCam.setObjectName("btnOnCam")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(890, -10, 111, 41))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(890, 20, 101, 21))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 991, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        #Nhấn btn CHỌN ẢNH để chọn ảnh từ file
        self.btnIP.clicked.connect(self.chose_file) #chạy hàm chose_file
        #Nhấn btn NHẬN DẠNG để nhận dạng
        self.btnOP.clicked.connect(self.Identification) #CHạy hàm Indentification
        #Bật camera
        self.btnOnCam.clicked.connect(self.OnCam)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.labelNameProject.setText(_translate("MainWindow", "NHẬN DẠNG BIỂN BÁO GIAO THÔNG"))
        self.labelPath.setText(_translate("MainWindow", "Path:"))
        self.labelName.setText(_translate("MainWindow", "Tên:"))
        self.btnIP.setText(_translate("MainWindow", "Chọn ảnh"))
        self.btnOP.setText(_translate("MainWindow", "Nhận dạng"))
        self.btnOnCam.setText(_translate("MainWindow", "Open Camera"))
        self.label.setText(_translate("MainWindow", "Nguyễn Văn Thuận"))
        self.label_2.setText(_translate("MainWindow", "Nguyễn Tiến Thịnh"))
    
    #Hàm chọn file hình ảnh
    def chose_file(self):
        fileName = QFileDialog.getOpenFileName()
        path = fileName[0]
        if path:
            self.labelIP.setPixmap(QtGui.QPixmap(path)) #Xuất ảnh ra
            self.labelPath.setText(path) #Xuất đường dẫn ra

    #Hàm nhận dạng
    def Identification(self):
        
        try:
           
            path=self.labelPath.text() #Lấy đường dẫn của hình ảnh
            image=cv2.imread(path) #Đọc hình ảnh
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
                    self.labelName.setText("Tên: "+result['Name'][id])
                    self.labelOP.setPixmap(QtGui.QPixmap(result['Path'][id]))
                    break
        except:
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Chưa chọn ảnh")    
            x=msg.exec_()
           
    #Hàm nhận dạng 2
    def Identification2(self):
        
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
    
    
   # -------------------------------------------------------
    def returnRedness(img):
    	yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    	y,u,v=cv2.split(yuv)
    	return v

    def threshold(img,T=150):
    	_,img=cv2.threshold(img,T,255,cv2.THRESH_BINARY)
    	return img 

    def findContour(img):
    	contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    	return contours

    def findBiggestContour(contours):
    	m = 0
    	c = [cv2.contourArea(i) for i in contours]
    	return contours[c.index(max(c))]

    def boundaryBox(img,contours):
    	x,y,w,h=cv2.boundingRect(contours)
    	img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    	sign=img[y:(y+h) , x:(x+w)]
    	return img,sign
    #-------------------------------------
    #Hàm bật camera để lấy hình ảnh        
    def OnCam(self):  
        #url = "http://192.168.1.4:4747/video"
        url = "http://192.168.43.1:4747/video"
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
        global clicked
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
                 
                self.Identification2()
               
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
        self.labelIP.setPixmap(QtGui.QPixmap('E:/Python/trafficsignsCNN/camera.png'))
        self.labelPath.setText('E:/Python/trafficsignsCNN/camera.png')
        #self.Identification()
        
        #-------------------------------------
        """
        cameraCapture = cv2.VideoCapture(0) 
        cv2.namedWindow('camera')
        cv2.setMouseCallback('camera', onMouse)
        
        # Read and process frames in loop
        success, frame = cameraCapture.read()  
        global clicked
        clicked = False       
        while success and not clicked:
            cv2.waitKey(1)
            success, frame = cameraCapture.read()    
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.medianBlur(gray, 37)
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                                      1, 500, param1=50, param2=40)      
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
                    image = frame[y-r:y+r, x-r:x+r]     
                cv2.imwrite('E:/Python/trafficsignsCNN/camera.png', image)    
                for i in circles[0, :]:
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.imshow('camera', frame)   
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        cameraCapture.release()
        self.labelIP.setPixmap(QtGui.QPixmap('E:/Python/trafficsignsCNN/camera.png'))
        self.labelPath.setText('E:/Python/trafficsignsCNN/camera.png')
        self.Identification()
        """
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

