#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model
import pandas as pd
import numpy as np
from random import seed
from random import choice
from random import randint


# In[2]:


model = load_model('/task3/hyperparameter.h5')


# In[3]:


import pandas as pd
file = open("labtest.csv")
numline = len(file.readlines())
print (numline)
file.close()

#get accuracy
columns = ['layers','filter_size','kernel_size','pooling','fc','neurons','epochs','acc']
dataset = pd.read_csv('labtest.csv',names=columns)
y = dataset['acc'].tolist()

Y = list(map(lambda x:x[2:-2], y))
Y=(Y[numline-1])
Y=float(Y)
acc_t = 0.001


# In[4]:


sequence=[]
for i in range (1,11):
    j= pow(2,i)
    sequence.append(j)


# In[5]:



while acc_t<Y:
    layers=1
    filt=choice(sequence)
    ker=randint(1, 10)
    pool=randint(1, 10)
    fc=randint(1, 5)
    fc_layer=choice(sequence)
    epo=randint(1, 25)
    x = [layers,filt,ker,pool,fc,fc_layer,epo]
    X = np.array(x).reshape(1,7)
    acc = model.predict(X)
    print(acc)
    if (acc<1):
        acc_t = acc
        with open("labtest.csv","a+") as f:
            f.write(f"{layers},{filt},{ker},{pool},{fc},{fc_layer},{epo},{acc}\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




