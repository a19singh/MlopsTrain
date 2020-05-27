#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[2]:


from keras import backend


# In[ ]:





# In[3]:


with open('input.txt', 'r') as file:
   lines = file.readlines()


# In[4]:


layer=int(lines[0])
filt=int(lines[1])
ker=int(lines[2])
pool=int(lines[3])
fc=int(lines[4])
fc_layer=int(lines[5])
epo=int(lines[6])


# In[5]:


dataset = mnist.load_data('mymnist.db')


# In[6]:


train , test = dataset


# In[7]:


X_train , Y_train = train


# In[ ]:





# In[8]:


X_test , Y_test =test


# In[9]:


if backend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)


# In[10]:


#X_train_id = X_train.reshape(-1 , 28*28)


# In[11]:


X_train =X_train.astype('float32')


# In[12]:


from keras.utils.np_utils import to_categorical


# In[13]:


y_train_cat = to_categorical(Y_train)


# In[14]:


from keras.layers import Convolution2D


# In[15]:


from keras.layers import MaxPooling2D


# In[16]:


from keras.layers import Flatten


# In[17]:


from keras.models import Sequential


# In[18]:


model = Sequential()


# In[19]:


for i in range(1,layer+1):
    model.add(Convolution2D(filters=filt,kernel_size=(ker,ker), activation='relu' , input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(pool,pool)))
    model.add(Flatten())


# In[ ]:





# In[20]:


from keras.layers import Dense


# In[21]:


for i in range(1,fc+1):
    model.add(Dense(units=fc_layer , activation='relu'))    


# In[22]:


model.add(Dense(units=10, activation='softmax'))


# In[23]:


model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])


# In[24]:


model.fit(X_train,y_train_cat ,epochs=epo)


# In[25]:


#X_train.shape


# In[26]:


y_test_cat = to_categorical(Y_test)
score = model.evaluate(X_test, y_test_cat, verbose=0)


# In[27]:


#Loss
# score[0]


# In[28]:


# Accuracy

x=float(score[1])
x=x*100
print(x)

# model.summary
# 

# In[30]:





# In[37]:


feed = open("data.txt","a+")
#feed.write("<----------------------------------->\r\n")
feed.write("Convol layer : %d\r\n" %layer)
feed.write("Filters : %d\r\n" %filt)
feed.write("kernels : %d\r\n" %ker)
feed.write("Pool size : %d\r\n" %pool)
feed.write("Fully connected layer : %d\r\n" %fc)
feed.write("Neurons : %d\r\n" %fc_layer)
feed.write("Epochs : %d\r\n" %epo)
feed.write("Accuracy : %f\r\n" %x)
feed.close()

# In[ ]:




