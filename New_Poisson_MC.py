import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import integrate
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
import time

start_time = time.time()
# Load completeness fraction
data = np.loadtxt('best_lum_func_new.txt') 
Lbins = np.array(data[:,0]) #Luminosity bins
dN = np.array(data[:,1]) #Mean LF

# Load completeness fraction
data = np.loadtxt('mass_list.txt')
mass = np.array(data[:,0]) #mass

#invert CDF of the LF for uniform sampling
dN_int = integrate.cumtrapz(dN, Lbins, initial=0)
dN_ICDF = UnivariateSpline(dN_int, Lbins, s=0.0)

#dwarf galaxy data
dwarf = ['SegueI','UrsaMajorII','ReticulumII','WillmanI','ComaBerenices','TucanaII','BootesI','IndusI','UrsaMinor','Draco','Sculptor','Sextans','HorologiumI','PhoenixII','UrsaMajorI','Carina','Hercules','Fornax','LeoIV','CanesVenaticiII','CanesVenaticiI','LeoII','LeoI','EridanusII', 'Grus_II','Tucana_III','Columba_I','Tucana_IV','Reticulum_III','Indus_II']

#number of MC iterations
NMC = 1000000
NMSP = np.trapz(dN,Lbins)
dSph_msp = NMSP*10.0**mass
pois_msp = np.random.poisson(dSph_msp,size=(NMC,len(dSph_msp)))
for i in range(len(dSph_msp)):
    print dwarf[i]

test_vals = np.random.uniform(0.0,np.amax(dN_int),size=NMC)
Lum_test = dN_ICDF(test_vals)

'''
Cum_Lum = np.zeros((NMC,len(dSph_msp)))
for i in range(len(dSph_msp)):
    for j in range(NMC):
        if pois_msp[j,i]>0.0:
            #Randomly Sample Luminosities
            CDF_vals = np.random.uniform(0.0,np.amax(dN_int),size=int(pois_msp[j,i]))
            Lum_vals = dN_ICDF(CDF_vals)
            Cum_Lum[j,i] = np.sum(Lum_vals) 
    label = 'new_dwarf_list/dwarf_text/%s.txt' % dwarf[i]
    np.savetxt(label, Cum_Lum[:,i])
    print dwarf[i]

job_time = time.time() - start_time
print(job_time)
'''
'''
#Create Plot
minorLocator = AutoMinorLocator()

bins = np.linspace(31,36,51)

fig, ax = plt.subplots()
plt.figure(1)
plt.hist(np.log10(Lum_test), bins=bins, normed=False, color='g', alpha=0.5, linewidth=1.0)
#plt.title('MSP Luminosity Function')
plt.xlabel('$L_\gamma (erg\cdot s^{-1})$')
plt.ylabel('Counts Per Bin')
ax.set_yscale('log')
#ax.set_xscale('log')
plt.axis((31.0,36.0,0.1,1e6))

#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')

plt.show()
'''
