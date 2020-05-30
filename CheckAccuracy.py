#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
file = open("labtest.csv")
numline = len(file.readlines())
print (numline)
file.close()

columns = ['layers','filter_size','kernel_size','pooling','fc','neurons','epochs','acc']
dataset = pd.read_csv('labtest.csv',names=columns)
y = dataset['acc'].tolist()

Y = list(map(lambda x:x[2:-2], y))
#get accuracy
#accuracy=float(row[7])
Y=(Y[numline-1])
Y=float(Y)
if (Y>=0.98):
    print("acquired")
else:
    print("not acquired")


# In[ ]:





# In[ ]:




