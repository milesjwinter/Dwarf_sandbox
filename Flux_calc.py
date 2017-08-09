import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import integrate
from scipy.stats import norm
import time

#Function to calculate flux for each dwarf
def flux_calc(Lbin, dN, mass, Dist):
    N = Lbins*dN*mass
    lum = np.trapz(N,Lbins)
    flux = lum/(4.0*np.pi*(Dist*3.0857E+21)**2)
    return flux

def lum_calc(Lbin, dN, mass):
    N = Lbins*dN*mass
    lum = np.trapz(N,Lbins)
    return lum

# Load Luminosity function
data = np.loadtxt('best_lum_func.txt') 
Lbins = np.array(data[:,0]) #Luminosity bins
dN = np.array(data[:,1]) #Mean LF
dNl = np.array(data[:,2]) #Lower statistical LF
dNh = np.array(data[:,3]) #Upper statistical LF
dSl = np.array(data[:,4]) #Lower systematic LF
dSh = np.array(data[:,5]) #Upper systematic LF

#LogL = 27.1711622458
#LogSL = 0.18046077219

# Load Poisson fluctuations
data = np.loadtxt('flux_results_test.txt')
dNPois = np.array(data[:,0]) #Poisson fluctuations mean
dPoisL = np.array(data[:,1]) #Poisson fluctuations unc.
dPoisU = np.array(data[:,2]) #Poisson fluctuations unc.

Poisn = dNPois - dPoisL
Poisp = dPoisU - dNPois
print 'Pois values'
for i in range(len(Poisn)):
    print Poisn[i], Poisp[i]

data = np.loadtxt('dwarf_stats.txt')
Dist = np.array(data[:,0])
mass  = 10.0**np.array(data[:,1])

Unc_array = np.zeros((30,5))
print 'Mean'
for j in range(len(mass)):
    lum_N = lum_calc(Lbins, dN, mass[j])
    Flux_N = flux_calc(Lbins, dN, mass[j], Dist[j])
    Flux_Nl = flux_calc(Lbins, dNl, mass[j], Dist[j])
    Flux_Nh = flux_calc(Lbins, dNh, mass[j], Dist[j])
    Flux_Sl = flux_calc(Lbins, dSl, mass[j], Dist[j])
    Flux_Sh = flux_calc(Lbins, dSh, mass[j], Dist[j])
    #UNl = np.fabs((Flux_N-Flux_Nl)/Flux_N)/np.log(10.0)
    #UNh = np.fabs((Flux_N-Flux_Nh)/Flux_N)/np.log(10.0)
    #USl = np.fabs((Flux_N-Flux_Sl)/Flux_N)/np.log(10.0)
    #USh = np.fabs((Flux_N-Flux_Sh)/Flux_N)/np.log(10.0)
    UNl = np.fabs(Flux_N-Flux_Nl)
    UNh = np.fabs(Flux_N-Flux_Nh)
    USl = np.fabs(Flux_N-Flux_Sl)
    USh = np.fabs(Flux_N-Flux_Sh)
    Unc_array[j,:] = np.array([Flux_N, UNl, UNh, USl, USh])
    #print USl, USh
    #print Flux_N
    print lum_N
#np.savetxt('flux_values.txt',Unc_array)

'''
#Create Plot
minorLocator = AutoMinorLocator()

bins = np.logspace(26,34,151)

fig, ax = plt.subplots()
plt.figure(1)
plt.hist(Lum, bins=bins, histtype='stepfilled', normed=False, color='b', alpha=1.0, linewidth=1.0)
#plt.title('MSP Luminosity Function')
plt.xlabel('$L_\gamma (erg\cdot s^{-1})$')
plt.ylabel('Counts Per Bin')
#ax.set_yscale('log')
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
'''
