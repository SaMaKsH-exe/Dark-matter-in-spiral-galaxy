#!/usr/bin/env python3 # -*- coding: utf-8 -*-

"""
Gravitaional Dynamics and Galaxy Formation - Rotation Curve of a Galaxy
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from astropy.io import fits
from scipy.optimize import curve_fit

plt.close('all')

#Loading 21 cm line data
data_21 = np.loadtxt('/Users/samakshkaushik/Desktop/Projects/Dark-matter-in-spiral-galaxy/UGC914/21cm.data')

velocity_21 = data_21[:,0]
flux_21 = data_21[:,1]+2.1


#Global constants
lmbd_em = 6564.61
pixel_scale = 0.189 # arcsec per pixel
kpc_per_arc = 0.16142854 #kpc per arcsec at the redshift of the galaxy
c = 3.0 * 10 ** 5 #speed of light [km/s]
G = 6.674 * 10**(-11) #NW gravitaional constant in SI units
#G = G * 6.44623408 * 10**(-7) # ((km / s)^2) * (kpc * (solar masses^(-1)))
binSize = 15 #size of bins
ncenter = 710 #435row number for the centre of the galaxy


####################################
####  FUNCTION DEFINITIONS      ####                       
####################################

def Redshift(lmbd_obs):    
    z = np.zeros_like(lmbd_obs)  
    for i in range(len(lmbd_obs)):      
        z[i] = (lmbd_obs[i] - lmbd_em) / lmbd_em  
    return z

def CiruclarSpeedExponentialDisk(y):

    vcsq = 4*np.pi*G*sigma0*Rd*y**2*\
    (scipy.special.i0(y)*scipy.special.k0(y)-\
     scipy.special.i1(y)*scipy.special.k1(y))

    return vcsq


#Load wavelength measurements 
data = np.loadtxt('/Users/samakshkaushik/Desktop/Projects/Dark-matter-in-spiral-galaxy/UGC914/lambda_ugc914.data')
afstand15 = np.array(data[:,0])
peaket15 = np.array(data[:,1])
usikkerhed15 = np.array(data[:,2])
newafstand15frac = [(i*binSize-ncenter)*pixel_scale*kpc_per_arc for i in afstand15] #skaleret afstand
newusikkerhed15 = np.array([2*np.sqrt(i) for i in usikkerhed15])#2sigma usikkerhed=95%konfidensinterval


#så laver vi usikkerhed på afstand
newafstand15low=[(i*binSize-ncenter)*pixel_scale*kpc_per_arc for i in afstand15]
newafstand15high=[((i+1)*binSize-ncenter)*pixel_scale*kpc_per_arc for i in afstand15]

newafstand15 = np.zeros((2,len(newafstand15low)))
newafstand15 [0,:] = newafstand15low
newafstand15 [1,:] = newafstand15high

newnewafstand15 = np.mean(newafstand15,axis=0) #gns afstand
usikkerhedafstand15= np.array(np.std(newafstand15,ddof=1,axis=0))#sigma på afstand
newusikkerhedafstand15 = [2*i for i in usikkerhedafstand15] #2sigma på afstand = 95% konfidensinterval


#beregner hastighed
znew = Redshift(peaket15)
znew_err = newusikkerhed15 / lmbd_em
vnew = c * znew + 24.8
vnew_err = c * znew_err


#First plot: lambda as a function of distance
plt.figure()
plt.grid()
plt.errorbar(newnewafstand15,peaket15,yerr=newusikkerhed15,xerr=newusikkerhedafstand15,fmt='.')
plt.ylabel('lambda i Å')
plt.xlabel('afstand i kpc')
plt.show()
#

#Second plot: rotation curve together with 21 cm data
fig, ax1 = plt.subplots()
plt.grid()
ax1.set_xlabel('afstand i kpc')
ax1.set_ylabel('hastighed i km/s')
ax1.errorbar(newnewafstand15,vnew,yerr=vnew_err,xerr=newusikkerhedafstand15,fmt='.')
ax1.set_ylim(2100, 2600)

ax2 = ax1.twiny()
ax2.set_xlabel('21cm Flux (ukendte enheder)')
ax2.set_ylabel('hastighed i km/s')
ax2.plot(flux_21,velocity_21,color='tab:red')
plt.show()

#Third plot: rotation curve together with theoretical models

#The stellar mass is determined from the SG4 survey (Spitzter imaging in the 3.6
#and 4.5 micron bands):

M_stellar = 10 ** 9.967 #[M_sun]
M_gas = 9*10**9 #HI region in [M_sun]
M_baryon = M_stellar + M_gas
M_sun = 1.989*10**30 #Mass of sun in [kg]
Rd = 4.73*3.086*10**19 # disk scale length

sigma0 = M_baryon * M_sun/ (2*np.pi*Rd**2) 
x = np.arange(0.001,15,0.1)
y = x*3.086*10**19/(2*Rd)
vcsq = CiruclarSpeedExponentialDisk(y)
vc = np.sqrt(vcsq)/1000 #km/s
vc_r = vc*np.sin(70/360*2*np.pi) # correct for inclination

plt.figure()
plt.grid()
plt.errorbar(newnewafstand15,vnew-(2329),yerr=vnew_err,xerr=newusikkerhedafstand15,fmt='.')
plt.ylabel('hastighed i km/s')
plt.xlabel('afstand i kpc')
plt.plot(x,vc_r,color='tab:red')
plt.plot(-x,-vc_r,color='tab:red')


plt.show()



