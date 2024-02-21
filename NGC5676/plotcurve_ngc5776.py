#!/usr/bin/env python3 # -*- coding: utf-8 -*-

"""
Rotation Curve of a Galaxy
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from astropy.io import fits
from scipy.optimize import curve_fit

plt.close('all')

#Global constants
lmbd_em =  6564.61 # H-alpha wavelenght in AA
pixel_scale = 0.189 # arcsec per pixel
c = 3.0 * 10 ** 5 #speed of light [km/s]


#Load wavelength measurements 
data = np.loadtxt('lambda.data')
afstandc = np.array(data[:,0])
sigma_afs = afstandc/afstandc*0.5
lambdac = np.array(data[:,1])
sigma_lamb = np.array(data[:,2]) 


#beregner hastighed
z = lambdac/lmbd_em-1
zmean = np.mean(z)
vel = (z-zmean)/(1+zmean)*c
sigma_vel = sigma_lamb/lambdac*c 

#First plot: lambda as a function of distance
plt.figure()
plt.grid()
plt.errorbar(afstandc,vel,xerr=sigma_afs,yerr=sigma_vel,fmt='.')
plt.ylabel('velocity i km/s')
plt.xlabel('afstand i kpc')
plt.show()




