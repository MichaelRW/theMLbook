# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:06:30 2020

@author: Michael R. Wirtzfeld
"""

#%% Environment

from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.pyplot.close("all")



#%% Example Reference

# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a


#%% Download the MINST Digit Dataset

mnist = fetch_openml('mnist_784')
#
# 70,000 hand-written digit images of size 28-by-28.


#%% Establish Training and Testing Datasets from Working Dataset

X_train, X_test, y_train, y_test = \
    train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)


#%% Show Illustrative Training Images and Respective Labels
    
plt.figure(figsize=(20,4))

for index, (image, label) in enumerate(zip(X_train[0:5], y_train[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: %s\n' % label, fontsize = 20)
    
    
#%% Instantiate Instance of the LogisticRegression Class with Default Parameters
    
model_logisticRegression = \
    LogisticRegression(penalty='l2', dual=False, \
    tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, \
    class_weight=None, random_state=None, solver='lbfgs', max_iter=1000, \
    multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
#
model_logisticRegression.fit(X_train, y_train)
yPredicted = model_logisticRegression.predict(X_test)


#%% Performance Evaluation - Confusion Matrix

confusionMatrix = metrics.confusion_matrix(y_test, yPredicted)
print(); print(confusionMatrix)


#%% Visualize Confusion Matrix using Seaborn

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(confusionMatrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
# plt.tight_layout()
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')


#%% Accuracy

score = model_logisticRegression.score(X_test, y_test)
print(); print('Accuracy score: {0:0.4f}'.format(score))


#%% Show Subset of Misclassified Digits

index = 0; misclassifiedIndexes = []

for label, predict in zip(y_test, yPredicted):
    if label != predict:
        misclassifiedIndexes.append(index)
        index +=1
        
plt.figure(figsize=(20,4))

for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.imshow(np.reshape(X_test[badIndex], (28,28)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(yPredicted[badIndex], y_test[badIndex]), fontsize = 15)

#%% Clean-up

print('\n\n*** Processing Complete ***\n\n')


