import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import integrate
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

#dwarf galaxy data
dwarf = ['SegueI','UrsaMajorII','ReticulumII','WillmanI','ComaBerenices','TucanaII','BootesI','IndusI','UrsaMinor','Draco','Sculptor','Sextans','HorologiumI','PhoenixII','UrsaMajorI','Carina','Hercules','Fornax','LeoIV','CanesVenaticiII','CanesVenaticiI','LeoII','LeoI','EridanusII', 'Grus_II','Tucana_III','Columba_I','Tucana_IV','Reticulum_III','Indus_II']

#number of MC iterations
NMC = 20000

for j in range(len(mass)):
    N = Lbins*dN*10.0**(mass[j])
    Lum = np.zeros(NMC)
    for i in range(NMC):
        PdNf = np.random.poisson(N,size=len(Lbins))
        Lum[i] = np.trapz(PdNf,Lbins)
        
    label = 'dwarf_list/dwarf_text/%s.txt' % dwarf[j]
    np.savetxt(label, Lum)
    print dwarf[j]

job_time = time.time() - start_time
print(job_time)

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
