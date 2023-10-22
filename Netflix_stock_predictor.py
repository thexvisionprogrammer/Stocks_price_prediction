#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from matplotlib import pyplot as plt


# In[4]:


from sklearn import model_selection


# In[5]:


from sklearn.metrics import confusion_matrix


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


from sklearn.preprocessing import MinMaxScaler


# In[9]:


from keras.models import Sequential


# In[10]:


import keras 


# In[11]:


from keras.models import Sequential


# In[12]:


from keras.layers import Dense


# In[13]:


from keras.layers import LSTM


# In[14]:


from keras.layers import Dropout


# In[15]:


df = pd.read_csv("NFLX.csv")


# In[16]:


df.head(10)


# # Training And Testing Of Data

# In[17]:


df.shape


# In[18]:


df1_train = df.reset_index()['Close']


# In[19]:


df1_train.head()


# In[20]:


df1_train.shape


# In[21]:


plt.plot(df1_train)


# # Scalling the data to make it fit

# In[22]:


sc = MinMaxScaler(feature_range = (0,1))


# In[23]:


df1_train = sc.fit_transform(np.array(df1_train).reshape(-1,1))


# In[24]:


df1_train


# # train-test split using cross validation 

# In[25]:


train_size = int(len(df1_train)*0.65)
test_size = len(df1_train)-train_size
train_data=df1_train[0:train_size,:]
test_data = df1_train[train_size:len(df1_train),:1]


# In[26]:


train_size, test_size


# # data preprocessing

# In[27]:


import numpy
def create_ds(dataset, time_step=1):
    x,y = [], []
    for i in range(len(dataset)-time_step-1):
        #i = 0...100
        a = dataset[i:(i+time_step), 0]
        x.append(a)
        y.append(dataset[i+time_step,0])
    return numpy.array(x), numpy.array(y)


# In[28]:


time_step = 100
x_train, y_train = create_ds(train_data, time_step)
x_test, y_test = create_ds(test_data, time_step)


# In[29]:


print(x_train.shape), print(y_train.shape)


# In[30]:


print(x_test.shape), print(y_test.shape)


# # reshape input to be (samples, time_steps, features) which is rquired for LSTM

# In[31]:


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)


# # lstm model building
# 

# In[32]:


algo = Sequential()
algo.add(LSTM(units = 50, return_sequences = True, input_shape = (100,1)))
algo.add(LSTM(units = 50, return_sequences = True))
algo.add(LSTM(units = 50))
algo.add(Dense(units = 1))
algo.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[33]:


algo.summary()


# In[34]:


algo.fit(x_train, y_train, validation_data = (x_test,y_test),epochs = 50, batch_size =64, verbose = 1)


# # prediction

# In[35]:


train_pred=algo.predict(x_train)
test_pred=algo.predict(x_test)


# # transform back to original form

# In[36]:


train_pred = sc.inverse_transform(train_pred)
test_pred = sc.inverse_transform(test_pred)


# # calculate RMSE perfomance metric

# In[37]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_pred))


# In[38]:


#test data RMSE
math.sqrt(mean_squared_error(y_test,test_pred))


# In[39]:


look_back = 100
trainpredictplot = numpy.empty_like(df1_train)
trainpredictplot[:, :] = np.nan
trainpredictplot[look_back:len(train_pred)+ look_back, :] = train_pred
#shift test prediction for plotting
testpredictplot = numpy.empty_like(df1_train)
testpredictplot[:, :] = np.nan
testpredictplot[len(train_pred)+ (look_back*2)+1:len(df1_train)-1, :] = test_pred
#plot baseline and prediction
plt.plot(sc.inverse_transform(df1_train))
plt.plot(trainpredictplot)
plt.plot(testpredictplot)
plt.show()


# In[40]:


len(test_data)


# In[41]:


x_input = test_data[1666:].reshape(1,-1)
x_input.shape


# In[42]:


temp_input = list(x_input)
temp_input = temp_input[0].tolist()


# In[43]:


from numpy import array

list_output = []
n_steps = 100
i=0
while(i<30):
    if(len(temp_input)>100):
        #print(temp_input)
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = algo.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        #print(temp_input)
        list_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = algo.predict(x_input, verbose =0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        list_output.extend(yhat.tolist())
        i=i+1
        
print(list_output)


# In[44]:


day_new = np.arange(1,101)
day_pred = np.arange(101,131)


# In[45]:


import matplotlib.pyplot as plt


# In[46]:


len(df1_train)-100


# In[47]:


plt.plot(day_new, sc.inverse_transform(df1_train[4944:]))
plt.plot(day_pred, sc.inverse_transform(list_output))


# In[48]:


df3 = df1_train.tolist()
df3.extend(list_output)
plt.plot(df3[4000:])


# In[49]:


df3 = sc.inverse_transform(df3).tolist()
plt.plot(df3)


# # This way we can predict the future stock prices
