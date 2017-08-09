import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator

############################################################
#Custom plot parameters
params = {
    #'backend': 'eps',
    'axes.labelsize': 24,
    #'text.fontsize': 12,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'xtick.major.size': 3,      # major tick size in points
    'xtick.minor.size': 1.5,    # minor tick size in points
    'xtick.major.size': 3,      # major tick size in points
    'xtick.minor.size': 1.5,    # minor tick size in points
    'text.usetex': True,
    #'axes.xmargin': 1.,
    #'figure.figsize': fig_size,
    'font.family':'serif',
    'font.serif':'Computer Modern Roman',
    'font.size': 24
    }
plt.rcParams.update(params)
############################################################

#Pull in data from text file
data = np.loadtxt('comp_unc_new.txt')
Lum_bins_new = np.array(data[:,0])
Mean_new = np.array(data[:,2]) 
Upper = np.array(data[:,3])
Lower = np.array(data[:,1])

data = np.loadtxt('comp_unc_bins.txt')
Lum_bins = np.array(data[:,0])
Mean = np.array(data[:,1])
Sigma = np.array(data[:,2])

Flux = UnivariateSpline(Lum_bins_new, Mean_new, s=0.0)
Inverted_Flux = UnivariateSpline(Mean_new, Lum_bins_new, s=0.0)
Unc_upper = UnivariateSpline(Lum_bins_new, Upper, s=0.0)
Unc_lower = UnivariateSpline(Lum_bins_new, Lower, s=0.0)
Avg_val = np.zeros(8)

for q in range(8):
    A = Lum_bins_new[q*10]
    B = Lum_bins_new[q*10+9]
    Int_val = quad(Flux, A, B)
    Avg_val[q] = Int_val[0]/(B-A)
Lum_val = Inverted_Flux(Avg_val)
Unc_val_upper = Unc_upper(Lum_val)
Unc_val_lower = Unc_lower(Lum_val)
data_points = np.transpose(np.array([Lum_val, Unc_val_lower, Avg_val, Unc_val_upper]))
print data_points
low = np.logspace(31.5,35.0,8)
high = np.logspace(32.0,35.5,8)
print Flux(np.logspace(31.5,35.5,10))

for i in range(8):
    print Lum_val[i]-low[i], '    ', high[i]-Lum_val[i]



fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.13)
plt.figure(1)
#minorLocator = AutoMinorLocator()
plt.plot(Lum_bins_new, Flux(Lum_bins_new), 'r', linewidth=2.0)
#plt.plot(Lum_bins, Mean,'b', linewidth=2.0)
plt.fill_between(Lum_bins_new, Unc_lower(Lum_bins_new), Unc_upper(Lum_bins_new), alpha=0.5, facecolor='grey')
#plt.plot(Lum_bins, 100.0*(Flux(Lum_bins)+Unc(Lum_bins)), 'b', linewidth=2.0)
#plt.plot(Lum_bins, 100.0*(Flux(Lum_bins)-Unc(Lum_bins)), 'g', linewidth=2.0)
#plt.title('Estimated Completeness vs $\gamma$-Ray Luminosity',fontsize=18)
plt.xlabel('$L_\gamma$ (erg/s)', labelpad=10)
plt.ylabel('Completeness Fraction', labelpad=10)

#Format tick marks
#ax.xaxis.set_minor_locator(minorLocator)
#ax.yaxis.set_minor_locator(minorLocator)
#plt.tick_params(which='major', length=7, width=2)
#plt.tick_params(which='minor', length=4)
#plt.minorticks_on()
plt.xscale('log')
plt.yscale('log')
#plt.grid('off')
plt.xlim([Lum_bins.min(),Lum_bins.max()])
plt.show()

