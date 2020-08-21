# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:06:30 2020

@author: Michael R. Wirtzfeld
"""

#%% Environment

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})



from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

matplotlib.pyplot.close("all")


#%% Example Reference

# https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python


#%% Load the Pima Indians Diabetes Dataset

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("./10 Datasets/pima-indians-diabetes.csv", header=None, names=col_names)
pima = pima.drop([0])  # Drop row with the original column labels.

print(pima.head())


#%% Select the Working Feature Columns and the Labelled/Target Column

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
features = pima[feature_cols]

labels = pima.label


#%% Establish Training and Testing Datasets from Working Dataset

X_train, X_test, y_train, y_test = \
    train_test_split( features, labels, test_size=0.25, random_state=0 )
    
    
#%% Instantiate Instance of LogisticRegression Class with Default Arguments
    
model_logisticRegression = \
    LogisticRegression(penalty='l2', dual=False, \
    tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, \
    class_weight=None, random_state=None, solver='lbfgs', max_iter=1000, \
    multi_class='auto', verbose=0, warm_start=False, n_jobs=None, \
    l1_ratio=None )
#
model_logisticRegression.fit(X_train,y_train)
y_pred = model_logisticRegression.predict(X_test)


#%% Performance Evaluation - Confusion Matrix

confusionMatrix = metrics.confusion_matrix(y_test, y_pred)
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


#%% Compute Performance Statistics

averagePrecision = metrics.accuracy_score(y_test, y_pred)
print(); print('Average precision-recall score: {0:0.4f}'.format(averagePrecision))

# True Positives (tp): 36
# True Negatives (tn): 118
# False Positives (fp): 12
# False Negatives (fn): 26
#
precisionScore = metrics.precision_score(y_test, y_pred, pos_label='1')
#
# tp / (tp + fp) = 36 / (36 + 12) = 0.7500
#
recallScore = metrics.recall_score(y_test, y_pred, pos_label='1')
#
# tp / (tp + fn) = 36 / (36 + 26) = 0.5806

print(); print('Precision: {0:0.4f}, Recall: {1:0.4f}'.format(precisionScore, recallScore))


#%% Receiver-Operator Curve (ROC) 

yPredictionProbability = model_logisticRegression.predict_proba(X_test)[::, 1]
falsePositveRate, truePositiveRate, thresholds = \
    metrics.roc_curve(y_test, yPredictionProbability, pos_label='1')
    
auc = metrics.roc_auc_score(y_test, yPredictionProbability)

print(); print('AUC: {0:0.4f}'.format(auc))

plt.figure();
plt.plot(falsePositveRate, truePositiveRate, label='Dataset, AUC='+str(auc))
plt.grid()
plt.show()
plt.axes().set_aspect('equal')


#%% Clean-up

print('\n\n*** Processing Complete ***\n\n')


