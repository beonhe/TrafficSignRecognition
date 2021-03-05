#Define
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image as img
import os
#Khai báo 

 
height = 30
width = 30
data=[]
labels=[]
channels = 3
n_inputs = height * width*channels

classes = 43
# Đọc dữ liệu từ DataTrain
for i in range(classes) :
    path = "E:/Python/trafficsignsCNN/DataTrain/{0}/".format(i)
    print(path)
    Class=os.listdir(path) #lấy tên file ảnh
    for a in Class:
        try:
            image=cv2.imread(path+a)
            image_from_array = img.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image)) #Thêm ảnh vào data
            labels.append(i) #Đánh dấu nhãn tương ứng với ảnh
        except AttributeError:
            print(" ")
            
Cells=np.array(data)
labels=np.array(labels)
 
#Xáo hỉnh ảnh chọn ngẫu nhiên đầu vào
s=np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]

#Phân chia dữ liệu

(X_train,X_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]
X_train = X_train.astype('float32')/255 
X_val = X_val.astype('float32')/255
(y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]
 
#Using one hote encoding for the train and validation labels
from keras.utils import to_categorical
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

#Khởi tạo model CNN keras

#import thư viện
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
#Khởi tạo model
model = Sequential()
#Thêm Conv Layer
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
#Pooling layer MaxPool
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
#Fully Connected Layer ReLu
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
#Fully Connected Layer softmax
model.add(Dense(43, activation='softmax'))

#Huấn luyện model
#Compilation of the model
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

#using ten epochs for the training and saving the accuracy for each epoch
epochs = 10
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,validation_data=(X_val, y_val))


#Lấy dữ liệu từ DataTest qua file CSV
#Predicting with the test data
y_test=pd.read_csv("E:/Python/trafficsignsCNN/Test.csv")
labels=y_test['Path'].to_numpy()
y_test=y_test['ClassId'].values
 
data=[]

for f in labels:
    image=cv2.imread('E:/Python/trafficsignsCNN/DataTest/'+f.replace('Test/', ''))
    image_from_array = img.fromarray(image, 'RGB')
    size_image = image_from_array.resize((height, width))
    data.append(np.array(size_image))
 

X_test=np.array(data)
X_test = X_test.astype('float32')/255 
pred = model.predict_classes(X_test)

#Accuracy with the test data
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)



# load json and create model
"""
json_file = open('E:/Python/trafficsignsCNN/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("E:/Python/trafficsignsCNN/model.h5")
"""
#Lưu model

#model save
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#Test


#testing 
"""
data=[]
image=cv2.imread('E:/Python/Testing/1.jpg')
#image=cv2.imread('E:/Python/trafficsignsCNN/DataTest/00015.png')
image_from_array = img.fromarray(image, 'RGB')
size_image = image_from_array.resize((height, width))
data.append(np.array(size_image))
 
X_test=np.array(data)
X_test = X_test.astype('float32')/255 
pred = model.predict_classes(X_test)


print(f"1: {pred[0]}")

cv2.imshow(f"{pred[0]}",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


#Convert to tflite
"""
TF_LITE_MODEL_FILE_NAME = "E:/Python/trafficsignsCNN/tf_lite_model.tflite"

tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = tf_lite_converter.convert()

tflite_model_name = TF_LITE_MODEL_FILE_NAME
open(tflite_model_name, "wb").write(tflite_model)
"""























