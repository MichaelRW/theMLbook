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

import matplotlib.pyplot as plt
import numpy as np

matplotlib.pyplot.close("all")



#%% Example Reference

# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a


#%%



#%% Function Definition

def f(x):
    """ Function to approximate by polynomial interpolation. """
    return 0.5 * x



#%% Create Signal with Noise

x_plot = np.linspace(-10, 10, 100)

# Generate points and keep a subset of them.
rng = np.random.RandomState(0)  # Set RNG state to replicate results.

x = np.linspace(-10, 10, 100)
rng.shuffle(x)  # Randomize order of elements in vector x.
x = np.sort(x[:10])  # Select first 10 elements.
#
noise = np.random.normal( loc=0.0, scale=1.0, size=len(x) )
y = f(x) + noise

# Create matrix versions of these arrays.
X = x[:, np.newaxis]  # Column vector -> 10-by-1
X_plot = x_plot[:, np.newaxis]  # Column vector -> 100-by-1

colors = ['red', 'red'];  lw = 2


#%% Regression Modeling

type_of_regression = ["Linear Regression", "Regression of Degree 10"]
fit = ["Fit", "Overfit"]

print()

for count, degree in enumerate( [1, 10] ):  # Starting enumeration at zero.
    
    plt.figure(count)
    #
    axes = plt.gca()
    axes.set_xlim([-15,15]); axes.set_ylim([-20,20])
    plt.scatter(x, y, color='navy', s=30, marker='o', label="Training Examples")
    plt.xticks([-10.0, -5.0, 0.0, 5.0, 10.0]); plt.yticks([-15.0,-10.0, -5.0, 0.0, 5.0, 10.0,15.0])
    
    # model = make_pipeline( PolynomialFeatures(degree), Ridge() )
    #
    # https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html
    #
    # Uses linear least squares with L2-norm (also known as Tikhonov Regularization)
    
    model = make_pipeline( PolynomialFeatures(degree), LinearRegression() )
    #
    # https://scikit-learn.org/stable/modules/linear_model.html
    #
    # Uses ordinary least squares.
    
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    #
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,label=type_of_regression[count])
    plt.legend(loc='best'); plt.grid()
    fig1 = plt.gcf()
    #
    yTrue = y; yPredict = model.predict(X)
    SMSE = mean_squared_error( yTrue, yPredict, sample_weight=None, multioutput='uniform_average', squared=True)
    print(f"Squared MSE: {SMSE}")
    
    # fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
    # fig1.savefig('linear-regression-' + fit[count] + '.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
    # fig1.savefig('linear-regression-' + fit[count] + '.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
    # fig1.savefig('linear-regression-' + fit[count] + '.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)

plt.show()


#%% Clean-up

print('\n\n*** Processing Complete ***\n\n')


