# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 09:22:07 2022

@author: ANEH
"""

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

gc_data = pd.read_csv(r"C:\Users\ANEH\Documents\Deep Learning Class\germanCredit.csv", sep=' ', header=None)

#%%
#Perform preprocessing
#Only want value of 0 and 1 for label
gc_data[20] = gc_data[20] - 1

#%%
gc_features = gc_data.copy()
gc_labels = gc_features.pop(20)

#%%
#Onehot encode all categorccal features
gc_features = pd.get_dummies(gc_features)

#%%
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

SEED = 12345
x_train, x_test, y_train, y_test = train_test_split(gc_features,gc_labels,test_size=0.3,random_state=SEED)
x_train_np = np.array(x_train)
x_test_np = np.array(x_test)

#%%
standardizer = sklearn.preprocessing.StandardScaler()
standardizer.fit(x_train_np)

x_train = standardizer.transform(x_train_np)
x_test = standardizer.transform(x_test_np)

#Data preparation is completeds

#%%
model = tf.keras.Sequential()

nClass = len(np.unique(np.array(y_test)))
# Normalization/ standard scaler done in previous cell
model.add(tf.keras.layers.InputLayer(input_shape=(x_train_np.shape[1],)))
model.add(tf.keras.layers.Dense(1024,activation='relu'))
model.add(tf.keras.layers.Dense(512,activation='relu'))
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(32,activation='relu'))
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(nClass,activation='softmax'))

#model = tf.keras.Model(inputs=inputs,outputs=outputs,name='gc_model')
model.summary()


#%%
model.compile(optimizer='adam' ,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=16,epochs=100)
#%%
import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = history.epoch 

plt.plot(epochs, training_loss, label='Training loss')
plt.plot(epochs, val_loss, label='validation loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.figure()

plt.plot(epochs, training_accuracy, label='Training accuracy')
plt.plot(epochs, val_accuracy, label='validation accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.figure()
plt.show()