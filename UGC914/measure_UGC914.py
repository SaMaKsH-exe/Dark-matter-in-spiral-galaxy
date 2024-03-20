#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Astronomiens fundament 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from matplotlib.widgets import Button

plt.close('all')

# hvilket bin vi skal gausfitte
n = 30

# Load data
UGC914 = fits.open(
    '/Users/samakshkaushik/Desktop/Projects/Dark-matter-in-spiral-galaxy/UGC914/UGC914_spec.fits', ignore_missing_end=True)
header = UGC914[0].header
data = UGC914[0].data

# Global constants
ncol = len(data[0])  # number of coloumns in the data
lmbd_em = 6564.61  # H-alpha wavelenght in AA
pixel_scale = 0.189  # arcsec per pixel
kpc_per_arc = 0.16142854  # kpc per arcsec at the redshift of the galaxy
binSize = 15  # size of bins
bins = len(data)/binSize  # number of bins
ncenter = 740  # 435row number for the centre of the galaxy
c = 3.0 * 10 ** 5  # speed of light [km/s]
G = 6.674 * 10**(-11)  # NW gravitaional constant in SI units
# G = G * 6.44623408 * 10**(-7) # ((km / s)^2) * (kpc * (solar masses^(-1)))

# Read the wavelength calibration information
cdelt1 = header['CDELT1']  # Increment, i.e. AA/pixel
crval1 = header['CRVAL1']  # Startwavelength in AA

wl = np.arange(crval1, ncol*cdelt1+crval1, cdelt1)  # wavlength array in AA


flux = []  # Placeholder for the summed spectrum in each bin of size 20
spec = []  # Placeholder for the flux as function of wavelength
lmbd_obs = []  # Placeholder for the measured wavelength
lmbd_err = []
distUpperBound = []
distLowerBound = []
dist = []
perr = []  # Placeholder for parameter error used in the curve_fit function


x = np.array_split(data, bins)  # spatial division in bin size

goodRegion = np.array([6500, 6700])  # good region is between 6500 and 6700 AA
# region converted to a range as an array of indicies
goodRange = np.where((wl >= goodRegion[0]) & (wl <= goodRegion[1]))

# defining the noise region
noiseRegion = np.array([[goodRegion[0], goodRegion[0]+50],
                       [goodRegion[1]-50, goodRegion[1]]])
# All indicies of noise in the good region to the left side of the peak
noiseRangeLeft = np.where((wl >= noiseRegion[0][0]) &
                          (wl <= noiseRegion[0][1]))
noiseRangeRight = np.where((wl >= noiseRegion[1][0]) &
                           (wl <= noiseRegion[1][1]))

###############################################################################

####################################
####  FUNCTION DEFINITIONS      ####
####################################

"""
Function shifts the flux signal to baseline 0 in order to
properly fit the emission line
"""


def BaseLineZero(flux):

    meanNoiseLeft = np.mean(flux[noiseRangeLeft])
    meanNoiseRight = np.mean(flux[noiseRangeRight])
    meanNoise = (meanNoiseLeft + meanNoiseRight) / 2
    newFlux = flux[goodRange] - meanNoise  # Flux at baseline 0
    return newFlux


"""
Function calculates the spread of the noise on both sides of ranges
"""


def NoiseSpread(flux):

    sigmaNoiseLeft = np.std(flux[noiseRangeLeft]) * \
        np.sqrt(len(flux[noiseRangeLeft]))

    sigmaNoiseRight = np.std(flux[noiseRangeRight]) * \
        np.sqrt(len(flux[noiseRangeRight]))

    sigmaNoise = (sigmaNoiseLeft + sigmaNoiseRight) * 0.5

    return sigmaNoise


"""
Gaussian function later used to fit the emission line
"""


def gauss_function(x, I, mu, sigma):

    #    return I/np.sqrt(2.0*np.pi)/sigma*np.exp(-0.5*((x-mu)/sigma)**2)
    return I/(np.sqrt(2.0*np.pi)*sigma)*np.exp(-0.5*((x-mu)/sigma)**2)

###############################################################################

####################################
####     BINNING DATA           ####
####################################


for i in range(len(x)):  # Sum flux in each bin of size 20

    flux.append(np.sum(x[i], axis=0))

for i in range(len(x)):  # Join together flux and spectrum

    spec.append(np.vstack((flux[i], wl)))


###############################################################################

####################################
####     MAIN                   ####
####################################


newFlux = BaseLineZero(flux[n])  # fratræk baggrunden
sigmaNoise = NoiseSpread(flux[n])
newsigmaNoise = NoiseSpread(flux[n])  # støj

# initial parameters has to be given the 'curve_fit' function
newinitialParameters = [np.max(newFlux), 6617, 20]  # startgæt til gaussfit
val, cov = curve_fit(
    gauss_function, wl[goodRange], newFlux, p0=newinitialParameters)
# giver optimerede værdier for lambda samt usikkerheder

# Create the figure and axes
fig, ax = plt.subplots()

# Plot initial data
line_data, = ax.plot(wl[goodRange], newFlux, '.', mec='k', label='Data')
line_fit, = ax.plot(np.linspace(6550, 6680, 1000), gauss_function(np.linspace(6550, 6680, 1000), *val), '-', c='#fc5a50',
         label='Gaussian fit')
ax.set_ylabel('Signal')
ax.set_xlabel('Wavelength [Å]')
ax.legend(loc='upper left', fancybox=True, shadow=True, framealpha=1,
           facecolor='#d8dcd6', edgecolor='black', prop={'size': 8}, markerscale=2)
ax.set_xlim(6550, 6680)

# Define the "Next" and "Previous" button positions
button_next_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
button_prev_ax = plt.axes([0.81, 0.05, 0.1, 0.075])

# Create the "Next" and "Previous" buttons
button_next = Button(button_next_ax, 'Next')
button_prev = Button(button_prev_ax, 'Previous')

# Function to handle "Next" button click
def next_clicked(event):
    global n
    n = min(n + 1, len(x) - 1)
    update_plot()

# Function to handle "Previous" button click
def prev_clicked(event):
    global n
    n = max(n - 1, 0)
    update_plot()

# Connect button click events to their respective functions
button_next.on_clicked(next_clicked)
button_prev.on_clicked(prev_clicked)

# Function to update plot
def update_plot():
    newFlux = BaseLineZero(flux[n])
    val, cov = curve_fit(gauss_function, wl[goodRange], newFlux, p0=newinitialParameters)
    line_data.set_ydata(newFlux)
    line_fit.set_ydata(gauss_function(np.linspace(6550, 6680, 1000), *val))
    ax.set_title(f"Bin {n}")  # Update title
    plt.draw()

plt.show()
