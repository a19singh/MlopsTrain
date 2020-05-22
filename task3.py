#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.datasets import mnist


# In[ ]:


from keras import backend


# In[ ]:


dataset = mnist.load_data('mymnist.db')


# In[ ]:


train , test = dataset


# In[ ]:


X_train , Y_train = train


# In[ ]:


#X_train.shape


# In[ ]:


X_test , Y_test =test


# In[ ]:


if backend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)


# In[ ]:


#X_train_id = X_train.reshape(-1 , 28*28)


# In[ ]:


X_train =X_train.astype('float32')


# In[ ]:


from keras.utils.np_utils import to_categorical


# In[ ]:


y_train_cat = to_categorical(Y_train)


# In[ ]:


from keras.layers import Convolution2D


# In[ ]:


from keras.layers import MaxPooling2D


# In[ ]:


from keras.layers import Flatten


# In[ ]:


from keras.models import Sequential


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Convolution2D(filters=32,kernel_size=(2,2), activation='relu' , input_shape=input_shape))


# In[ ]:





# In[ ]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[ ]:


model.add(Flatten())


# In[ ]:


from keras.layers import Dense


# In[ ]:


model.add(Dense(units=128 , activation='relu'))


# In[ ]:


model.add(Dense(units=10, activation='softmax'))


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(X_train,y_train_cat ,epochs=25)


# In[ ]:


#X_train.shape


# In[ ]:


y_test_cat = to_categorical(Y_test)
score = model.evaluate(X_test, y_test_cat, verbose=0)


# In[ ]:


#Loss
# score[0]


# In[ ]:


# Accuracy
print(score[1])


# In[ ]:




