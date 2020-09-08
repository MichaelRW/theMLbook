# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 11:24:48 2020

@author: Michael R. Wirtzfeld
"""

#%% Environment

from __future__ import print_function
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})

import matplotlib.pyplot as plt
import numpy as np
import time


#%% Function Definitions

def plot_data_and_bestLine(x, y, w, b):
    
    xValues = np.linspace(0, max(x), num=100); estimatedY = predict(xValues, w, b)

    plt.figure()
    
    plt.scatter(x, y, color='#1f77b4', marker='o')
    plt.plot(xValues, estimatedY, color='r', linewidth=2.0)

    plt.xlabel("Advertising Spending (Millions of Dollars)")
    plt.ylabel("Sales (Units)")
    plt.title("Sales Versus Advertising Spending")
    plt.grid()
    # fig1 = plt.gcf()
    # fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
    
    
def plot_lossHistory(lossHistory):
    
    plt.figure()
    
    plt.plot(lossHistory, color='b', linewidth=2.0)
    
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss History")
    plt.title("Loss Versus Training Epoch")
    plt.grid()
    

def update_w_and_b(spendings, sales, w, b, alpha):
    '''
    Parameters
    ----------
    spendings : numpy.ndarray
        Number of dollars spent on advertising for a given medium (millions of dollars).
    sales : numpy.ndarray
        Corresponding sales value (units).
    w : numpy.float64
        Slope of fitted linear regression line.
    b : numpy.float64
        Vertical axis intercept of fitted linear regression line.
    alpha : float
        Fixed learning rate for batch gradient decent method.

    Returns
    -------
    w : numpy.float64
        Updated value of slope for fitted linear regression line.
    b : numpy.float64
        Update value of vertical axis intercept for fitted linear regression line.

    '''
    
    dr_dw = 0.0; dr_db = 0.0; observations=len(spendings)

    for i in range(observations):
        dr_dw += -2 * spendings[i] * (sales[i] - (w * spendings[i] + b))
        dr_db += -2 * (sales[i] - (w * spendings[i] + b))

    w = w - (dr_dw/float(observations)) * alpha
    b = b - (dr_db/float(observations)) * alpha

    return w, b


def train(spendings, sales, w, b, alpha, epochs):
    
    startTime = time.time()
    
    lossHistory = np.zeros(shape=int(epochs), dtype=float)
    
    print()
    
    for e in range(int(epochs)):
        w, b = update_w_and_b(spendings, sales, w, b, alpha)

        if (e % 100 == 0):            
            print('Epoch: {0}, Loss: {1:0.4f}'.format(e, loss(spendings, sales, w, b)))
            
        lossHistory[e] = loss(spendings, sales, w, b)
        
    print(); print('Training time: {0:0.2f} seconds'.format(time.time()-startTime))
            
    return w, b, lossHistory


def trainingViaSKLearn(spendings, sales):
    
    model = LinearRegression().fit(spendings, sales)
    
    return model
    

def loss(spendings, sales, w, b):
    
    N = len(spendings); total_error = 0.0
    
    for i in range(N):
        total_error += (sales[i] - (w*spendings[i] + b))**2
        
    return total_error / N


def predict(x, w, b):
    
    return w*x + b


#%% Processing
    
index, tv, radioAdvertisingSpending, newspaper, sales = \
    np.loadtxt("./10 Datasets/Advertising.txt", delimiter= "\t", unpack = True, skiprows=1)
    
x=radioAdvertisingSpending; y=sales

w, b, lossHistory = train(x, y, w=0.0, b=0.0, alpha=5e-4, epochs=15.0e3)

plot_data_and_bestLine(x, y, w, b)
plot_lossHistory(lossHistory)
#
x_new = 23.0
print(); print('New X: {0}, Estimated Sales: {1:0.2f}'.format(x_new, predict(x_new, w, b)))


# Via Scikit Learng Library
model = trainingViaSKLearn(x.reshape(-1,1), y.reshape(-1,1)); yPredicted=model.predict([[x_new]])
print(); print(yPredicted)


#%% Clean-up

print('\n\n*** Processing Complete ***\n\n')


