import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator

#Pull in data from text file
data = np.loadtxt('comp_unc_new.txt')
Lum_bins = np.array(data[:,0])
Mean = np.array(data[:,1]) 
#Sigma = np.array(data[:,2])
dNdL = UnivariateSpline(Lum_bins, 1.0/(Mean*Lum_bins), s=0.0)
Inverted_dNdL = UnivariateSpline(Mean*Lum_bins, Lum_bins, s=0.0)
Comp = UnivariateSpline(Lum_bins, Mean, s=0.0)
#Comp_unc = UnivariateSpline(Lum_bins, Sigma, s=0.0)
avg_val = np.zeros(8)

for q in range(8):
    A = Lum_bins[q*10]
    B = Lum_bins[q*10+9]
    Int_val = quad(dNdL, A, B)
    avg_val[q] = (B-A)/Int_val[0]

Lum_val = Inverted_dNdL(avg_val)
Comp_val = Comp(Lum_val)
#Comp_val_unc = Comp_unc(Lum_val) 
Lum_low = Lum_val-np.logspace(31.5,35.0,8)
Lum_high = np.logspace(32.0,35.5,8)-Lum_val
data_points = np.transpose(np.array([Lum_val, Lum_low, Lum_high, Comp_val]))
np.savetxt('avg_data_points_new.txt',data_points)
print data_points
#for i in range(8):
#    print Comp_val_unc[i]

'''
fig, ax = plt.subplots()
plt.figure(1)
minorLocator = AutoMinorLocator()
plt.plot(Lum_bins, 1.0/(Mean*Lum_bins))
#plt.plot(Lum_bins, dNdL(Lum_bins),'b')
#plt.plot(Lum_val, 1.0/(Lum_val*Comp_val), 'r*')
#plt.plot(Lum_bins, 100.0*Flux(Lum_bins), 'r', linewidth=2.0)
#plt.plot(Lum_bins, 100.0*(Flux(Lum_bins)+Unc(Lum_bins)), 'b', linewidth=2.0)
#plt.plot(Lum_bins, 100.0*(Flux(Lum_bins)-Unc(Lum_bins)), 'g', linewidth=2.0)
plt.title('Estimated Completeness vs $\gamma$-Ray Luminosity',fontsize=18)
plt.xlabel('$\log_{10}{L_\gamma} (erg\cdot s^{-1})$',fontsize=15)
plt.ylabel('Completeness ($\%$)',fontsize=15)

#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
ax.set_xscale('log')
ax.set_yscale('log')
plt.grid('off')
plt.show()
'''
