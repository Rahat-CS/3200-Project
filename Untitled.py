#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import  load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


dataset=r'D:\Face Mask Detection\dataset'
imagePaths=list(paths.list_images(dataset)) ## make a list of the datasets


# In[3]:


imagePaths


# In[4]:


print(len(imagePaths))


# In[5]:


data=[]
labels=[]

for i in imagePaths:
    label=i.split(os.path.sep)[-2]
    labels.append(label)
    image=load_img(i,target_size=(224,224))
    image=img_to_array(image)   ## after executing,size will be shown
    image=preprocess_input(image)  ## data will be shown
    data.append(image)


# In[6]:


data   ## it's in the list form


# In[7]:


labels


# In[8]:


data=np.array(data,dtype='float32')   ## convert into numpy array
labels=np.array(labels)


# In[9]:


data.shape


# In[10]:


labels   ## label with numpy array


# In[11]:


lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)


# In[12]:


labels


# In[13]:


train_X,test_X,train_Y,test_Y= train_test_split(data,labels,test_size=0.20,stratify=labels,random_state=10)


# In[14]:


train_X ## 80% training data


# In[15]:


train_Y  ##80% label


# In[16]:


train_X.shape   ##in output 3 is rgb


# In[17]:


train_Y.shape


# In[18]:


test_X ## rest 20%


# In[19]:


test_Y #rest 20%


# In[20]:


test_X.shape


# In[21]:


test_Y.shape


# In[22]:


aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=.2,height_shift_range=.2,shear_range=.15,horizontal_flip=True,vertical_flip=True,fill_mode='nearest')


# In[23]:


aug


# In[24]:


baseModel=MobileNetV2(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))


# In[25]:


baseModel.summary()


# In[26]:


headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name='Flatten')(headModel)
headModel=Dense(128,activation='relu')(headModel)  
headModel=Dropout(0.5)(headModel)   ##overfitting
headModel=Dense(2,activation='softmax')(headModel)  ##50% wearing musk assumption

model=Model(inputs=baseModel.input,outputs=headModel)


# In[27]:


for layer in baseModel.layers:
    layer.trainable=False


# In[28]:


model.summary()


# In[ ]:





# In[ ]:





# In[29]:


learning_rate=0.001
Epochs=20
BS=12

opt=Adam(lr=learning_rate,decay=learning_rate/Epochs)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

H=model.fit(
    aug.flow(train_X,train_Y,batch_size=BS),
    steps_per_epoch=len(train_X)//BS,
    validation_data=(test_X,test_Y),
    validation_steps=len(test_X)//BS,
    epochs=Epochs
)

model.save(r'D:\Face Mask Detection\dataset')


# In[30]:


predict=model.predict(test_X,batch_size=BS)
predict=np.argmax(predict,axis=1)
print(classification_report(test_Y.argmax(axis=1),predict,target_names=lb.classes_))


# In[ ]:





# In[ ]:




