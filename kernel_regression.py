# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:06:30 2020

@author: Michael R. Wirtzfeld
"""

#%% Environment

from sklearn.kernel_ridge import KernelRidge

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import sys  # sys.exit()

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 25})


#%% Function Definition

def f(x):
    """ Function to approximate by polynomial interpolation. """
    return x * (x)  # Quadratic polynomial.


#%% Generate quadratic signal plus noise (uniform between -5 and 5).

x_plot = np.linspace(-5, 2, 100)

x = np.linspace(-5, 2, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:50])

noise = [ (-5 + np.random.random()*5) for i in range(len(x)) ]
y = f(x) + noise

# plt.plot(x, y, 'b'); plt.plot(x, y, 'r.'); ax = plt.gca(); ax.set_aspect('equal'); ax.grid('True')


#%% Create matrix versions of signal arrays/vectors.

X = x[:, np.newaxis]  # Convert 1-dimensional array to 2-dimensional column array.
X_plot = x_plot[:, np.newaxis]  # Convert 1-dimensional array to 2-dimensional column array.


#%% Define Kernel Function

def kernel(x1, x2, b = 2):
    z = (x1 - x2) / b
    return (1/math.sqrt(2 * 3.14)) * np.exp(-z**2/2)


#%% Compute and Plot Kernel Regressions

colors = ['red', 'blue', 'orange']; lw = 2

fit = [ "Strong Overfit", "Weak Overfit", "Good"]

for count, degree in enumerate( [0.1, 0.5, 3] ):
    
    plt.figure(count)    
    axes = plt.gca(); axes.set_xlim([-5,2]); axes.set_ylim([-10,30])
    
    plt.scatter(x, y, color='navy', s=30, marker='o', label="training examples")
    
    model = KernelRidge(alpha=0.01, kernel=kernel, kernel_params = {'b':degree})
    model.fit(X, y)
    
    y_plot = model.predict(X_plot)
    
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="b = " + str(degree))
    plt.title("Fit Condition: " + fit[count])

    plt.legend(loc='upper right'); plt.grid()
    
    # fig1 = plt.gcf()
    # fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
    # fig1.savefig('../../Illustrations/kernel-regression-' + str(count) + '.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
    # fig1.savefig('../../Illustrations/kernel-regression-' + str(count) + '.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
    # fig1.savefig('../../Illustrations/kernel-regression-' + str(count) + '.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)


plt.show()


#%% Clean-up

print('\n\n*** Processing Complete ***\n\n')


