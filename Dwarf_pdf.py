import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import norm

#Function to read in data
def dwarf_info(i):
    dwarf = ['SegueI','UrsaMajorII','ReticulumII','WillmanI','ComaBerenices','TucanaII','BootesI','IndusI','UrsaMinor','Draco','Sculptor','Sextans','HorologiumI','PhoenixII','UrsaMajorI','Carina','Hercules','Fornax','LeoIV','CanesVenaticiII','CanesVenaticiI','LeoII','LeoI','EridanusII','Grus_II','Tucana_III','Columba_I','Tucana_IV','Reticulum_III','Indus_II']
    Dist = np.array([23.0,32.0,32.0,38.0,44.0,58.0,66.0,69.0,76.0,76.0,86.0,86.0,87.0,95.0,97.0,105.0,132.0,147.0,154.0,160.0,218.0,233.0,254.0,330.0,53.0,25.0,182.0,48.0,92.0,214.0])
    name = dwarf[i]
    label = 'dwarf_text/%s.txt' % name
    lum = np.loadtxt(label)
    #flux = np.log10(lum[lum>0.0]/(4.0*np.pi*(Dist[i]*3.0857E+21)**2))
    flux = lum/(4.0*np.pi*(Dist[i]*3.0857E+21)**2)
    return lum, flux, name

results = np.zeros((30,3))
for i in range(30): 
    Lum, Flux, Name = dwarf_info(i)
    results[i,:] = np.percentile(Flux, [50,16,84])
    print Name
'''
    
    #fbins = np.logspace(np.log10(Flux[Flux>0.0].min()),np.log10(Flux.max()),100)
    fbins = np.linspace(Flux.min(),Flux.max(),100)
    fig, ax = plt.subplots()
    plt.figure(1)
    minorLocator = AutoMinorLocator()
    plt.hist(Flux, bins=fbins, histtype='stepfilled', normed=1.0, color='b', alpha=1.0, linewidth=1.0)
    plt.title('Energy FLux PDF: %s' % Name, fontsize=18)
    plt.xlabel('Flux $(erg\cdot cm^{-2}\cdot s^{-1})$',fontsize=15)
    plt.ylabel('Counts Per Bin',fontsize=15)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    #plt.xlim([Flux[Flux>0.0].min(),Flux.max()])
    plt.tick_params(which='major', length=7, width=2)
    plt.tick_params(which='minor', length=4)
    plt.minorticks_on()
    plt.grid('off')
    plt.savefig('dwarf_upper_plots/%s_flux_up.pdf' % Name)
    plt.show()
    
'''
np.savetxt('flux_results_test.txt',results)

'''
fig, ax = plt.subplots()
plt.figure(1)
minorLocator = AutoMinorLocator()
plt.hist(Lum, bins=lbins, histtype='stepfilled', normed=False, color='b', alpha=1.0, linewidth=1.0)
plt.title('Luminosity PDF: %s' % Name,fontsize=18)
plt.xlabel('$L_\gamma (erg\cdot s^{-1})$',fontsize=15)
plt.ylabel('Counts Per Bin',fontsize=15)
#ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim([Lum.min(),Lum.max()])
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.show()
'''

