
# coding: utf-8

# In[1]:

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import keras
import fnmatch
import os
import numpy as np
import sklearn
from sklearn import metrics


# In[2]:

def fcount(path):
    """ Counts the number of files in a directory """
    count = 0
    for f in os.listdir(path):
        #print(f)
        if os.path.isfile(os.path.join(path, f)):
            if (f[-4:] == '.jpg'):
                count += 1
        else:
            count += fcount(os.path.join(path, f))

    return count


# In[ ]:




# In[3]:



def jpgnamessub(path):
    l = []
    #count = 0
    for f in os.listdir(path):
        #print(f)
        if os.path.isfile(os.path.join(path, f)):
            if (f[-4:] == '.jpg'):
                #count += 1
                l.append([path,f]) 
        else:
            l.extend(jpgnamessub(os.path.join(path, f)))

    return l

def jpgnames(path):
    l = []
    #count = 0
    for f in os.listdir(path):
        #print(f)
        if os.path.isfile(os.path.join(path, f)):
            if (f[-4:] == '.jpg'):
                #count += 1
                l.append([path,f]) 
        else:
            l.extend(jpgnamessub(os.path.join(path, f)))

    return l


# In[ ]:




# In[4]:

batch_size = 1


# In[5]:

basepath = '/Users/omogensen/galdata'
#basepath = '/home/kerasvm/nas/GalaxyZoo/gzdatatest'


# In[6]:

imagenames = jpgnames(basepath)


# In[7]:

numvalid = fcount(basepath + '/valid')
numtest = fcount(basepath + '/test')


# In[8]:

# create the base pre-trained model
base_model = InceptionV3(weights=None, include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='softmax')(x)
# this is the model we will train
model = Model(input=base_model.input, output=predictions)


# In[9]:

#def conf_matr(y_true,y_pred):
#    return sklearn.metrics.confusion_matrix(y_true,y_pred)


# In[10]:

#def metrictest(y_true,y_pred):
    #sklearn.metrics.confusion_matrix(y_true,y_pred)
#    return y_true


# In[11]:

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['categorical_accuracy' , 'accuracy'])


# In[ ]:




# In[12]:

test_datagen = ImageDataGenerator()


# In[13]:

subpath = '/valid'


# In[14]:

validation_generator = test_datagen.flow_from_directory(
        basepath + subpath,
        batch_size=batch_size,
        class_mode='categorical'
    ,shuffle=False)


# In[ ]:




# In[15]:

model.load_weights(basepath + '/top3_retrain.h5')


# In[ ]:




# In[16]:

predictions = model.predict_generator(validation_generator,steps=numvalid,max_q_size=100,workers=1,pickle_safe=False,verbose=0)


# In[17]:

print('save predictions')
np.save(basepath + '/predictions', predictions)
np.savetxt(basepath + "/predictions.csv", predictions, delimiter=",")


# In[18]:

numpred = np.argmax(predictions,1)


# In[ ]:




# In[19]:

length = len(basepath + subpath)
#images = sorted([a for a in imagenames if (a[0][0:length] == basepath + '/test')],key=lambda a: a[1])
images = [a for a in imagenames if (a[0][0:length] == basepath + subpath)]


# In[20]:

classes = sorted([a[len(basepath) + 1 + len(subpath):] for a in list(set([im[0] for im in images]))])
images = [[a[0][len(basepath) + 1 + len(subpath):],a[1]] for a in images]


# In[21]:

print(classes)


# In[22]:

numtrue = [classes.index(a[0]) for a in images]


# In[23]:

#numpred


# In[24]:

numpredtrue = np.array(numtrue,dtype=int)


# In[25]:

pred = np.zeros([numpred.shape[0],2],dtype=int)
for i in range(pred.shape[0]):
    pred[i,0] = numpred[i]
    pred[i,1] = numpredtrue[i]


# In[26]:

print('save predicted classes')
np.save(basepath + '/intpredictions', numpred)
np.savetxt(basepath + "/intpredictions.csv", numpred, delimiter=",")


# In[27]:

print('save true classes')
np.save(basepath + '/intpredictionstrue', numpredtrue)
np.savetxt(basepath + "/intpredictionstrue.csv", numpredtrue, delimiter=",")


# In[28]:

print('save int predictions (both)')
np.save(basepath + '/predtrue', pred)
np.savetxt(basepath + "/predtrue.csv", pred, delimiter=",")


# In[29]:

print('save file name order')
f = open(basepath + '/filenames.txt','w')
for i in range(len(images)):
    f.write(images[i][0] + ',' + images[i][1] + '\n')
f.flush()
f.close()


# In[30]:

#images


# In[31]:

print(metrics.confusion_matrix(numpredtrue,numpred))


# In[32]:

print(model.evaluate_generator(validation_generator,steps=numvalid,max_q_size=100,workers=1,pickle_safe=False))


# In[ ]:



