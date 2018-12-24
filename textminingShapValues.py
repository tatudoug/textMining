#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 13:37:39 2018

@author: douglas
"""

import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import linear_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import shap

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
print(twenty_train.target_names)
print(len(twenty_train.data))
print(len(twenty_train.filenames))
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])
twenty_train.target[:10]
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])


# count vect
count_vect = CountVectorizer(max_df=0.5, min_df=10, max_features=1000)
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)

print(count_vect.vocabulary_.get(u'algorithm'))
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts).toarray()
print( X_train_tf.shape)
print("a")
print(X_train_counts)
print("b")
print(X_train_tf)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tf, twenty_train.target)

y_pred = clf.predict(X_train_tf)
print( sum(y_pred==twenty_train.target)/len(y_pred) )

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

model = OneVsRestClassifier(SVC(random_state=0,probability=True))
model.fit(X_train_tf, twenty_train.target)

y_pred = model.predict(X_train_tf)
print('SVM', sum(y_pred==twenty_train.target)/len(y_pred) )

#explainer = shap.KernelExplainer(model.predict_proba,X_train_tf[0:1000,:])
explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_train_tf[0:1000,:],100))


shap_values = explainer.shap_values(X_train_tf[0,:])

#num_features = shap_values[0].shape[1]
shap.summary_plot(np.array(shap_values), X_train_tf[0,:], color_bar=True)

shap.force_plot(explainer.expected_value[0], shap_values[0], X_train_tf[0,:],feature_names=sorted(count_vect.vocabulary_.keys()),matplotlib=True)