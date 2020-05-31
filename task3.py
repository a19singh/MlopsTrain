#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras import backend
from keras.utils.np_utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from random import seed
from random import choice
from random import randint


dataset = mnist.load_data('mymnist.db')
(X_train , Y_train), (X_test , Y_test) = dataset

with open('input.txt', 'r') as file:
   lines = file.readlines()


layer=int(lines[0])
filt=int(lines[1])
ker=int(lines[2])
pool=int(lines[3])
fc=int(lines[4])
fc_layer=int(lines[5])
epo=int(lines[6])
    
#layer=1
#filt=32
#ker=4
#pool=4
#fc=1
#fc_layer=32
#epo=2
   

print(f"{layer},{filt},{ker},{pool},{fc},{fc_layer},{epo}")


    ## ---------------------------------------------


if backend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

X_train =X_train.astype('float32')
y_train_cat = to_categorical(Y_train)

model = Sequential()

model.add(Convolution2D(filters=filt,kernel_size=(ker,ker), activation='relu' , input_shape=input_shape, padding="same"))
model.add(MaxPooling2D(pool_size=(pool,pool)))



for i in range(1,layer):
    model.add(Convolution2D(filters=filt,kernel_size=(2,2), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
model.add(Flatten())


for i in range(1,fc+1):
    model.add(Dense(units=fc_layer , activation='relu'))


    # Output Layer
model.add(Dense(units=10, activation='softmax'))


    # Compile and fit
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train_cat ,epochs=epo)
print("[+] Done Fitting")

y_test_cat = to_categorical(Y_test)
score = model.evaluate(X_test, y_test_cat, verbose=0)

    # Accuracy

x=float(score[1])
# x=x*100
print(x)


    # Write to datasets
with open("realrecords.csv","a+") as f:
    f.write(f"{layer},{filt},{ker},{pool},{fc},{fc_layer},{epo},{x}\n")


# In[ ]:




