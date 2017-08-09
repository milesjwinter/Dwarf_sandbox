import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import norm
from matplotlib.font_manager import FontProperties

############################################################
#Function to read in data
def dwarf_info(i):
    dwarf = ['SegueI','UrsaMajorII','ReticulumII','WillmanI','ComaBerenices','TucanaII','BootesI','IndusI','UrsaMinor','Draco','Sculptor','Sextans','HorologiumI','PhoenixII','EridanusIII','UrsaMajorI','Carina','PictorisI','Hercules','Fornax','LeoIV','CanesVenaticiII','CanesVenaticiI','LeoII','LeoI','EridanusII']
    Dist = np.array([23.0,32.0,32.0,38.0,44.0,58.0,66.0,69.0,76.0,76.0,86.0,86.0,87.0,95.0,95.0,97.0,105.0,126.0,132.0,147.0,154.0,160.0,218.0,233.0,254.0,330.0])
    name = dwarf[i]
    label = 'dwarf_text/%s.txt' % name
    lum = np.loadtxt(label)
    flux = lum[lum>0.0]/(4.0*np.pi*(Dist[i]*3.0857E+21)**2)/2.984e-3
    return lum, flux, name

############################################################
def logHist(x, n_bins=35, **kwargs):
    log_x = np.log10(x)
    log_bins = np.linspace(np.percentile(log_x, 0.01),
                           np.percentile(log_x, 99.99),
                           n_bins)
    delta_log_x = (log_bins[-1] - log_bins[0]) / (n_bins - 1.)


    weights = np.tile(1. / (delta_log_x * len(x)), len(x))
    results = plt.hist(log_x, bins=log_bins, weights=weights,
                         histtype='stepfilled', **kwargs)
############################################################

Lum0, Flux0, Name0 = dwarf_info(0)
Lum1, Flux1, Name1 = dwarf_info(9)
Lum2, Flux2, Name2 = dwarf_info(10)
Lum3, Flux3, Name3 = dwarf_info(19)

params = {
    #'backend': 'eps',
    'axes.labelsize': 28,
    #'text.fontsize': 12,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'xtick.major.size': 3,      # major tick size in points
    'xtick.minor.size': 1.5,    # minor tick size in points
    'xtick.major.size': 3,      # major tick size in points
    'xtick.minor.size': 1.5,    # minor tick size in points
    'text.usetex': True,
    #'axes.xmargin': 1.,
    #'figure.figsize': fig_size,
    'font.family':'serif',
    'font.serif':'Computer Modern Roman',
    'font.size': 18
    }
plt.rcParams.update(params)

#Create Plot
fig, ax = plt.subplots()
plt.figure(1)
logHist(Flux0, color='g', label='Segue 1, $M_*=3.4\\times 10^{2}M_\odot$')
logHist(Flux1, color='b', label='%s, $M_*=3.2\\times 10^{5}M_\odot$' % Name1)
logHist(Flux2, color='r', label='%s, $M_*=3.9\\times 10^{6}M_\odot$' % Name2)
logHist(Flux3, color='m', label='%s, $M_*=2.5\\times 10^{7}M_\odot$' % Name3)

syst = np.array([(3.36-1.29)*10.0**(-15),(3.36+2.26)*10.0**(-15)])
# Axis tricks
plt.yscale('log')
plt.xlim(-17.0, -11.0)
plt.ylim(1.0e-3, 5.0e2)
fontP = FontProperties()
fontP.set_size(20.0)
#plt.ylim(1.e-4, 1.e2)
if False:
    # Label ticks in linear space
    x_ticks = np.arange(-4, 8, 2)
    x_labels = []
    for x_tick in x_ticks:
        x_labels.append('$10^{{{0:d}}}$'.format(x_tick))
    plt.xticks(x_ticks, x_labels)
    plt.xlabel('Luminosity (Units)', labelpad=10) # Linear-space ticks
else:
    # Label ticks in log space
    plt.xlabel('Flux $>500$ MeV (ph cm$^{-2}$ s$^{-1}$)') # Log-space ticks
plt.axvline(np.log10((3.36-0.93)*10.0**(-15)), color='k', lw=1.5, linestyle='dashed', label='LF statistical uncertainty')
plt.axvline(np.log10((3.36+1.42)*10.0**(-15)), color='k', lw=1.5, linestyle='dashed')
plt.axvline(np.log10((3.36-2.01)*10.0**(-15)), color='k', lw=1.5, linestyle='solid',label='LF systematic uncertainty')
plt.axvline(np.log10((3.36+5.16)*10.0**(-15)), color='k', lw=1.5, linestyle='solid')
plt.ylabel('Counts (arb. units)', labelpad=10)
plt.xticks([-17.0,-16.0,-15.0,-14.0,-13.0,-12.0,-11.0], ['$10^{-17}$', '$10^{-16}$', '$10^{-15}$', '$10^{-14}$', '$10^{-13}$', '$10^{-12}$', '$10^{-11}$'])
handles, labels = ax.get_legend_handles_labels()
plt.legend(reversed(handles), reversed(labels), loc=0, ncol=1, bbox_to_anchor=(0, 0, 1, 1),fancybox=False,shadow=False,framealpha=0.0)
plt.show()

'''
#lbins = np.logspace(np.log10(Lum.min()),np.log10(Lum.max()),100)
fbins0 = np.logspace(np.log10(Flux0.min()),np.log10(Flux0.max()),20)
fbins1 = np.logspace(np.log10(Flux1.min()),np.log10(Flux1.max()),20)
fbins2 = np.logspace(np.log10(Flux2.min()),np.log10(Flux2.max()),20)
fbins3 = np.logspace(np.log10(Flux3.min()),np.log10(Flux3.max()),20)

w0 = np.array([1.0/float(len(Flux0))]*len(Flux0))
w1 = np.array([1.0/float(len(Flux1))]*len(Flux1))
w2 = np.array([1.0/float(len(Flux2))]*len(Flux2))
w3 = np.array([1.0/float(len(Flux3))]*len(Flux3))

#fbins0 = np.linspace(Flux0.min(),Flux0.max(),20)
#fbins1 = np.linspace(Flux1.min(),Flux1.max(),20)
#fbins2 = np.linspace(Flux2.min(),Flux2.max(),20)
#fbins3 = np.linspace(Flux3.min(),Flux3.max(),20)
'''
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
'''
fig, ax = plt.subplots()
plt.figure(1)
minorLocator = AutoMinorLocator()
plt.hist(Flux0, bins=fbins0, histtype='stepfilled', normed=False, weights=w0, color='b', alpha=1.0, linewidth=1.0,label='%s' % Name0)
plt.hist(Flux1, bins=fbins1, histtype='stepfilled', normed=False, weights=w1, color='r', alpha=1.0, linewidth=1.0,label='%s' % Name1)
plt.hist(Flux2, bins=fbins2, histtype='stepfilled', normed=False, weights=w2, color='g', alpha=1.0, linewidth=1.0,label='%s' % Name2)
plt.hist(Flux3, bins=fbins3, histtype='stepfilled', normed=False, weights=w3, color='m', alpha=1.0, linewidth=1.0,label='%s' % Name3)
plt.title('Energy FLux PDF', fontsize=18)
plt.xlabel('Flux $(erg\cdot cm^{-2}\cdot s^{-1})$',fontsize=15)
plt.ylabel('Frequency (arb. units)',fontsize=15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
#plt.xlim([Flux.min(),Flux.max()])
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.legend(loc="upper left")
plt.grid('off')
plt.show()
'''
