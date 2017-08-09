#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import integrate

# data from text file
data = np.loadtxt('list_msp.txt') #Load data
Lum = np.array(data) # Mass from Data
"""
#Pull in data from text file
data = np.loadtxt('lum_bins.txt') #Load data
B = np.array(data[:,0]) # bin from Data
Bu = np.array(data[:,1]) # bin uncertanty from Data
S = np.array(data[:,2]) # sources from Data
Su = np.array(data[:,3]) # sources uncertanty from Data
"""
#Pull in data from text file
data = np.loadtxt('non_log_bins.txt') #Load data
B = np.array(data[:,0]) # bin from Data
Bn = np.array(data[:,1]) # bin uncertanty from Data
Bp = np.array(data[:,2]) # bin uncertanty from Data
S = np.array(data[:,3]) # sources from Data
Su = np.array(data[:,4]) # sources uncertanty from Data

#define Fit line
def f(L, k = 28.83, Lb = 10.0, a1 = -0.1993, a2 = 0.5594):
    dN = 0.0
    
    if (L <= Lb):
        dN = k*(L)**(-a1)

    else:
        dN = k*(Lb)**(a2-a1)*(L)**(-a2)   

    return dN

#Create Plot
minorLocator = AutoMinorLocator()

bins = np.linspace(31.5,35.5,9)

fig, ax = plt.subplots()
plt.figure(1)
#plt.hist(Lum, bins=bins, histtype='step', normed=False, color='k', alpha=1.0, linestyle=('dashed'), linewidth=1.5, label='incomplete')
plt.errorbar(B, B**2*S, xerr=[Bn,Bp], yerr=B**2*Su, fmt='or', linewidth=1.5, label='complete')
#plt.title('MSP Luminosity Function')
plt.xlabel('$\log_{10}{L_\gamma} (erg\cdot s^{-1})$')
plt.ylabel('Counts')
#plt.legend(loc="upper right")
ax.set_yscale('log')
ax.set_xscale('log')
#plt.axis((31.5,35.5,0.0001,5))

#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')

plt.show()

