#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:05:55 2019

@author: douglas
"""
import numpy as np

def inter(list1, list2):
    return np.array(list( set(list1).intersection(list2) ))


def uni(list1, list2):
    buff = np.r_[list1, list2]
    return np.array(list(set(buff)))
    
def accMod(y_prev, y_test): 
    # expectes a vector of 1 and 0 
    
    acc = 0.0 # inicializes the accuracy score
    
    for val in np.arange((y_prev.shape[0])):
        y_prev_I = [x for x in np.arange( (y_prev[val,:].shape[0]) )  if y_prev[val,x] ]
        y_test_I = [x for x in np.arange( (y_test[val,:].shape[0]) )  if y_test[val,x] ]
        
        uV = uni(y_prev_I,y_test_I)
        iV = inter(y_prev_I,y_test_I)
        acc += iV.shape[0]/uV.shape[0]
        
    return(acc/y_prev.shape[0])
  

lst1 = np.array([15, 9, 10, 56, 23, 78, 5, 4, 9] )
lst2 = np.array([9,9, 4, 5, 36, 47, 26, 10, 45, 87] )

k = inter(lst1,lst2)
u = uni(lst1,lst2)

#print(inter(lst1,lst2))
#print(uni(lst1,lst2))



lst1 = np.array([[0, 1, 1, 0, 1,0],[1, 0, 1, 0, 1,0]] )
lst2 = np.array([[0, 0, 1, 0, 1,0],[1, 0, 1, 0, 1,0]] )

accMod(lst1, lst2)


from sklearn.datasets import make_multilabel_classification

# this will generate a random multi-label dataset
X, y = make_multilabel_classification( n_labels = 20, allow_unlabeled = False)

# keras model



import tensorflow as tf
import keras.backend as K
import keras.backend as int_shape

def customLoss(yTrue,yPred):
    print(yTrue)
    print(yPred)
    
    yTrue = tf.cast(yTrue, tf.bool)
    yPred = tf.cast(tf.round(yPred), tf.bool)
    
    t = tf.logical_and(yTrue, yPred) # intersection
    j = tf.logical_or(yTrue, yPred) # union
    
    a = tf.reduce_sum(tf.cast(t,tf.float32), axis=1) # sum values in the line
    b = tf.reduce_sum(tf.cast(j,tf.float32), axis=1) # sum values in the line
    
    x = tf.divide(a,b)

    print('f',tf.reduce_mean(x))
    
    f = tf.reduce_mean(x)
    
  

    '''    
    acc = 0.0 # inicializes the accuracy score
    
    
    
    
    #get_shape().as_list()
    
    batch_size, n_elems = yPred.get_shape()
    
    for val in range( n_elems):
        
        try:
            n_elems2 = yPred[val,:].get_shape()
        except:
            n_elems2 = 100
            
        yPred_I = [x for x in range(n_elems2[0]) ] # if tf.cond(yPred[val,x],1) ]
        yTrue_I = [x for x in range(n_elems2[0]) ] #  if yTrue[val,x] ]
        
        uV = uni(yPred_I,yTrue_I)
        iV = inter(yPred_I,yTrue_I)
        acc += iV.shape[0]/uV.shape[0]
    '''        
    return(tf.reduce_mean(x))
    
    

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=X.shape[1], kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(y.shape[1], kernel_initializer='normal', activation='sigmoid'))

i = model.predict(X)

model.compile(loss=customLoss, optimizer='adam')

model.fit(X,y)

	# Compile model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	






