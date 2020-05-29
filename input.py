#!/usr/bin/env python
# coding: utf-8

# In[1]:


# getting row no. of last row
file = open("labtest.csv")
numline = len(file.readlines())
print (numline)
file.close()


# In[2]:


#reading all lines of the file
file_variable = open('labtest.csv')
all_lines_variable = file_variable.readlines()


# In[3]:


# extracting last row as a string
sequence =all_lines_variable[numline-1]


# In[4]:


#converting string to list
row=sequence.split(',')


# In[5]:


#add new inputs to the file:
with open("input.txt","w+") as f:
    for i in range (0,7):
        f.write("%d\n"%int(row[i]))


# In[ ]:




