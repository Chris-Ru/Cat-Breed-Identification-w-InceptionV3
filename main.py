import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from scipy import ndimage, misc
import skimage
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Sequential
from keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout



filelist  = []

for dirname, _, filenames in os.walk('/kaggle/input'):
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


df = pd.DataFrame( list( zip (Filepaths, labels) ), columns = ['Filepath', 'Labels'] )

df

from sklearn.utils import shuffle
df = (df.sample(frac = 1).reset_index()).drop(columns = 'index')
df


f,a = plt.subplots(nrows=4, ncols=3,figsize=(13, 7),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(a.flat):
    ax.imshow(plt.imread(df.Filepath[i]))
    ax.set_title(df.Labels[i])
    
plt.tight_layout()
plt.show()


ax=pd.value_counts(df['Labels'],ascending=True).plot(kind='barh', fontsize="40", title="Distribution Of classes", figsize=(15,8))
ax.set(xlabel="Images per class", ylabel="Classes")
ax.xaxis.label.set_size(40)
ax.yaxis.label.set_size(40)
ax.title.set_size(60)
plt.show()


df.Labels.value_counts()


train_ratio = .75
validation_ratio = 0.10
test_ratio = 0.25

train, test = train_test_split(df, test_size = test_ratio )
val, test = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio))


img_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2,height_shift_range=0.2, horizontal_flip = 'true')



x_train =  img_datagen.flow_from_dataframe(dataframe = train,  x_col='Filepath', y_col='Labels',  target_size=(299, 299), shuffle=False, batch_size=10, seed=10)
x_val = img_datagen.flow_from_dataframe(dataframe = val,  x_col='Filepath', y_col='Labels',  target_size=(299, 299), shuffle=False, batch_size=10, seed=10)
x_test = img_datagen.flow_from_dataframe(dataframe = test,  x_col='Filepath', y_col='Labels',  target_size=(299, 299), shuffle=False, batch_size=10, seed=10)


x_train


i_model = InceptionV3(weights= 'imagenet', include_top=False, input_shape=(299, 299, 3))

for layer in i_model.layers:
    layer.trainable = False
    
i_model.summary() 


predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
predictions


labels = x_train.class_indices
labels



test["Labels"].replace({"Bombay": 0,'British Shorthair': 1,
                        'Burmese': 2,
                        'Calico': 3,
                        'Dilute Calico': 4,
                        'Himalayan': 5,
                        'Munchkin': 6,
                        'Ragdoll': 7,
                        'Russian Blue': 8,
                        'Siberian': 9}, inplace = True)


test_accuracy = model.evaluate(x_test)[1] * 100
print('Test accuracy is : ',test_accuracy, '%' )



confusion_matrix(test.Labels , predictions)



from sklearn.metrics import accuracy_score, f1_score
print('F1 score is',f1_score(test.Labels, predictions, average = 'weighted'))


predicted_probab =model.predict_proba(x_test)
predicted_probab



print("ROC- AUC score is", roc_auc_score( test.Labels, predicted_probab, multi_class='ovr'))