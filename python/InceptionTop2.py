
# coding: utf-8

# In[2]:

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


# In[ ]:

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

batch_size = 100


# In[ ]:

basepath = '/Users/omogensen/galdata'


# In[ ]:

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
        basepath + '/train',  # this is the target directory
        batch_size=batch_size,
        class_mode='categorical')


# In[ ]:

numtrain = fcount(basepath + '/train')


# In[ ]:

test_datagen = ImageDataGenerator()


# In[ ]:

validation_generator = test_datagen.flow_from_directory(
        basepath + '/valid',
        batch_size=batch_size,
        class_mode='categorical')


# In[ ]:

test_generator = test_datagen.flow_from_directory(
        basepath + '/test',
        batch_size=batch_size,
        class_mode='categorical')


# In[ ]:

numvalid = fcount(basepath + '/valid')
numtest = fcount(basepath + '/test')


# In[ ]:


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)



# In[ ]:

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='softmax')(x)


# In[ ]:


# this is the model we will train
model = Model(input=base_model.input, output=predictions)


# In[ ]:



# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
    
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# In[ ]:


# train the model on the new data for a few epochs
model.fit_generator(
        train_generator,
        steps_per_epoch=numtrain // batch_size,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=numvalid // batch_size)


# In[ ]:

model.save_weights(basepath + '/top_layer_retrain.h5')


# In[ ]:


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)


# In[ ]:

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True


# In[ ]:

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')


# In[ ]:

cback = keras.callbacks.ModelCheckpoint(basepath + '/checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)


# In[ ]:



# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(
        train_generator,
        steps_per_epoch=numtrain // batch_size,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=numvalid // batch_size
        , callbacks=[cback])


# In[ ]:

print('save weights')
model.save_weights(basepath + '/top3_retrain.h5')


# In[ ]:

#model.load_weights(basepath + '/checkpoint.h5')


# In[ ]:

#model.evaluate_generator(validation_generator,steps=numvalid,max_q_size=100,workers=1,pickle_safe=False)


# In[ ]:
print('predict')
model.predict_generator(validation_generator,steps=numvalid,max_q_size=100,workers=1,pickle_safe=False,verbose=1)


# In[27]:

predictions = model.predict_generator(validation_generator,steps=numvalid,max_q_size=100,workers=1,pickle_safe=False,verbose=1)


# In[1]:
print('save predictions')
np.save(basepath + '/predictions', predictions)
np.savetxt(basepath + "/predictions.csv", predictions, delimiter=",")


# In[29]:

#predictions

