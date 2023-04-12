# 1. Reading Modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import cv2 as cv
import numpy as np
from scipy import ndimage, misc
import skimage
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Sequential
from keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout

from IPython.display import SVG, Image
#from livelossplot import PlotLossesTensorFlowKeras
from livelossplot import PlotLossesKeras
from livelossplot.keras import PlotLossesCallback

filelist  = []

for dirname, _, filenames in os.walk('kaggle\data'):
    for filename in filenames:
        filelist.append (os.path.join(dirname, filename))

len(filelist)

filelist

labels_needed = ['Bombay', 'Calico', 'Burmese', 'Himalayan', 'Munchkin', 'Ragdoll', 'Siberian', 'British Shorthair', 'Russian Blue', 'Dilute Calico']

Filepaths   = []
labels = []

for image_file in filelist:
    label = image_file.split(os.path.sep)[-2]
    if label in labels_needed:

        Filepaths.append(image_file)
        labels.append(label)


set(labels)

len(Filepaths), len(labels)

# Creating a dataframe with file paths and the labels for them

df = pd.DataFrame( list( zip (Filepaths, labels) ), columns = ['Filepath', 'Labels'] )
df

from sklearn.utils import shuffle
df = (df.sample(frac = 1).reset_index()).drop(columns = 'index')
df

f,a = plt.subplots(nrows=4, ncols=3,figsize=(13, 7),
                        subplot_kw={'xticks': [], 'yticks': []})

# Displaying first 12 pictures

f, a = plt.subplots(ncols=2,nrows=3, sharex=True, sharey=True)

for i, ax in enumerate(a.flat):
    ax.scatter([i//2+1, i],[i,i//3])
        
plt.tight_layout()
plt.show()

ax=pd.value_counts(df['Labels'],ascending=True).plot(kind='barh',
                                                       fontsize="40",
                                                       title="Distribution Of classes",
                                                       figsize=(15,8))
ax.set(xlabel="Images per class", ylabel="Classes")
ax.xaxis.label.set_size(40)
ax.yaxis.label.set_size(40)
ax.title.set_size(60)
plt.show()

# Checking for class imbalance

df.Labels.value_counts()

# Splitting the data And Creating data generator

train_ratio = .75
validation_ratio = 0.10
test_ratio = 0.25

train, test = train_test_split(df, test_size = test_ratio )
val, test = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio))


img_datagen = ImageDataGenerator(rescale=1./255, 
                                rotation_range=30, 
                                width_shift_range=0.2,
                                height_shift_range=0.2, 
                                horizontal_flip = 'true')
   

x_train =  img_datagen.flow_from_dataframe(dataframe = train,  x_col='Filepath', y_col='Labels',  target_size=(299, 299), shuffle=False, batch_size=30, seed = 12)
x_val = img_datagen.flow_from_dataframe(dataframe = val,  x_col='Filepath', y_col='Labels',  target_size=(299, 299), shuffle=False, batch_size=30, seed = 12)
x_test = img_datagen.flow_from_dataframe(dataframe = test,  x_col='Filepath', y_col='Labels',  target_size=(299, 299), shuffle=False, batch_size=30, seed = 12)


x_train

# Modelling

i_model = InceptionV3(weights= 'imagenet', include_top=False, input_shape=(299, 299, 3))


for layer in i_model.layers:
    layer.trainable = False
    
i_model.summary()


model = Sequential()
model.add(i_model)
model.add(GlobalAveragePooling2D())

model.add(Dense(32))
model.add(Dropout(0.20))
model.add(Dense(3, activation = 'softmax'))
model.summary()


model.compile(optimizer = SGD(),
             loss="categorical_crossentropy",
             metrics=["accuracy"])


len(x_train)


len(x_test)


len(x_val)


history = model.fit(x_train, 
                    validation_data = x_val,
                    steps_per_epoch = len(x_train),
                    validation_steps = len(x_val), 
                    epochs = 40, 
                    verbose = 2,callbacks=[PlotLossesKeras()])


# Predicting on test data

predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
predictions


labels = x_train.class_indices
labels


test["Labels"].replace({"Abyssinian": 0,'Balinese': 1,
 'Tonkinese': 2 }, inplace = True)

#Evaluating the test data

# Test Accuracy


test_accuracy = model.evaluate(x_test)[1] * 100
print('Test accuracy is : ',test_accuracy, '%' )

# Confusion Matrix

cf = confusion_matrix(test.Labels , predictions)
cf

# F1 Score

from sklearn.metrics import accuracy_score, f1_score
print('F1 score is',f1_score(test.Labels, predictions, average = 'weighted'))

predicted_probab =model.predict_proba(x_test)
predicted_probab


# ROC - AUC Score
predicted_probab =model.predict_proba(x_test)
predicted_probab

print("ROC- AUC score is", roc_auc_score( test.Labels, predicted_probab, multi_class='ovr'))