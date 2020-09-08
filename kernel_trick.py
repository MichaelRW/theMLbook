# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:06:30 2020

@author: Michael R. Wirtzfeld
"""

#%% Environment

from mpl_toolkits.mplot3d import Axes3D

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
# import sys  # sys.exit()

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})


#%% Function Definitions

def f_outer(x1):
    
    result = []  # Empty list.
    
    for x in x1:
        # np.random.seed(seed=0)
        side = random.uniform(0, 1);
        sq = math.sqrt(10 * 10 - x * x)
        
        if side > 0.5:
            sq = sq * (-1)
            
        result.append(sq)
        
    return np.asarray(result)


def f_inner(x1):
    
    result = []  # Empty list.
    
    for x in x1:
        # np.random.seed(seed=0)
        side = random.uniform(0, 1);
        sq = math.sqrt(3 * 3 - x * x)
        
        if side > 0.5:
            sq = sq * (-1)
            
        result.append(sq)
        
    return np.asarray(result)


#%% Generate Data

x_inner = np.linspace(-3, 3, 100); x_outer = np.linspace(-10, 10, 100)

rng = np.random.RandomState(0)  # Set seed for reproducibility.
rng.shuffle(x_inner); rng.shuffle(x_outer)

x_inner = np.sort(x_inner[:30]); x_outer = np.sort(x_outer[:30])

noise = [ (-1 + np.random.random()) for i in range(len(x_inner)) ]
y_inner = f_inner(x_inner) + noise

noize = [ (-1 + np.random.random()) for i in range(len(x_outer)) ]
y_outer = f_outer(x_outer) + noise

colors = ['blue', 'red']; lw = 2


#%% Transform Data using Non-linear Transform - Quadratic Mapping

type_of_regression = [ "linear regression", "regression of degree 10"] 
fit = [ "fit", "overfit" ]

plt.figure(1); axes = plt.gca(); axes.set_xlim([-11,11]); axes.set_ylim([-11,11]); plt.grid()
axes.set_aspect('equal')

plt.scatter(x_inner, y_inner, color='navy', s=30, marker='o')
plt.scatter(x_outer, y_outer, color='red', s=30, marker='o')

x_inner_transformed = np.asarray([x * x for x in x_inner])
y_inner_transformed = np.asarray([math.sqrt(2) * x * y for x, y in zip(x_inner, y_inner)])
z_inner_transformed = np.asarray([y * y for y in y_inner])

x_outer_transformed = np.asarray([x * x for x in x_outer])
y_outer_transformed = np.asarray([math.sqrt(2) * x * y for x, y in zip(x_outer, y_outer)])
z_outer_transformed = np.asarray([y * y for y in y_outer])

fig = plt.figure(2); ax = Axes3D(fig); ax.set_yticks([-75, 0, 75])

ax.scatter(x_inner_transformed, y_inner_transformed, z_inner_transformed, color='navy', marker='o')
ax.scatter(x_outer_transformed, y_outer_transformed, z_outer_transformed, color='red', marker='o')

ax.view_init(14, -32)

plt.show()


#%% Clean-up

print('\n\n*** Processing Complete ***\n\n')


