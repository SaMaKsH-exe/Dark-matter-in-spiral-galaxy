# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:07:33 2019

@author: chjor
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from scipy.optimize import curve_fit

plt.close('all')


#Loading 21 cm line data
data_21 = np.loadtxt('21cmwork/21cm.data')
velocity_21 = data_21[:,0]
flux_21 = data_21[:,1]+2.2

#defining busyfunction
def busy_func(x, a, b, c, d, e, f):
    return (a/4.) * ( scipy.special.erf(b*(d+(x-e)))+1.) * ( scipy.special.erf(c*(d-(x-e)))+1.) * (f*(x-e)**2+1)


guess = (11.,0.02,0.03,210.,2300.,0.001) #startguess
var,usik = curve_fit(busy_func,velocity_21,flux_21,p0=guess) #curvefitting busyfunction


#plotting
plt.figure()
#plt.plot(velocity_21[130:243],flux_21[130:243],'.', label = 'data')
#plt.plot(velocity_21[130:243],busy_func(velocity_21[130:243], *var),'-',label = 'fit')
plt.plot(velocity_21,flux_21,'.', label = 'data')
plt.plot(velocity_21,busy_func(velocity_21, *var),'-',label = 'fit')
plt.legend()
plt.show()
