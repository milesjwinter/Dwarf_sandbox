import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import norm
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
############################################################
#Custom plot parameters
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
fontP = FontProperties()
fontP.set_size('small')
############################################################

#Function to read in data
def dwarf_info(i):
    dwarf = ['SegueI','Tucana_III','ReticulumII','UrsaMajorII','WillmanI','ComaBerenices','Tucana_IV','Grus_II','TucanaII','BootesI','IndusI','Draco','UrsaMinor','Sculptor','Sextans','HorologiumI','Reticulum_III','PhoenixII','UrsaMajorI','Carina','Hercules','Fornax','LeoIV','CanesVenaticiII','Columba_I','Indus_II','CanesVenaticiI','LeoII','LeoI','EridanusII']
    #dwarf = ['SegueI','UrsaMajorII','ReticulumII','WillmanI','ComaBerenices','TucanaII','BootesI','IndusI','UrsaMinor','Draco','Sculptor','Sextans','HorologiumI','PhoenixII','UrsaMajorI','Carina','Hercules','Fornax','LeoIV','CanesVenaticiII','CanesVenaticiI','LeoII','LeoI','EridanusII','Grus_II','Tucana_III','Columba_I','Tucana_IV','Reticulum_III','Indus_II']
    #Dist = np.array([23.0,32.0,32.0,38.0,44.0,58.0,66.0,69.0,76.0,76.0,86.0,86.0,87.0,95.0,97.0,105.0,132.0,147.0,154.0,160.0,218.0,233.0,254.0,330.0,53.0,25.0,182.0,48.0,92.0,214.0])
    Dist = np.array([23.0,25.0,32.0,32.0,38.0,44.0,48.0,53.0,58.0,66.0,69.0,76.0,76.0,86.0,86.0,87.0,92.0,95.0,97.0,105.0,132.0,147.0,154.0,160.0,182.0,214.0,218.0,233.0,254.0,330.0])
    name = dwarf[i]
    label = 'dwarf_text/%s.txt' % name
    lum = np.loadtxt(label)
    flux = np.zeros(len(lum))
    for j in range(len(lum)):
        if lum[j]>0.0:
            flux[j] = np.log10(lum[j]/(4.0*np.pi*(Dist[i]*3.0857E+21)**2*2.984e-3))
        else:
            flux[j] = -20.0
    return lum, flux, name

results = np.zeros((30,3))
data = np.loadtxt('dSph_msp_list.txt')
NMSP = np.array(data) 

dSph_name = ['Segue 1','Tucana III','Reticulum II','Ursa Major II','Willman I','Coma Berenices','Tucana IV','Grus II','Tucana II','Bootes I','Indus I','Draco','Ursa Minor','Sculptor','Sextans','Horologium I','Reticulum III','Phoenix II','Ursa Major I','Carina','Hercules','Fornax','Leo IV','Canes Venatici II','Columba I','Indus II','Canes Venatici I','Leo II','Leo I','Eridanus II']

fore_data = np.loadtxt('flux_foreground.txt')
#Name = ['SegueI','Tucana_III','ReticulumII','UrsaMajorII','WillmanI','Coma_Berenices','Tucana_IV','Grus_II','TucanaII','BootesI','IndusI','Draco','UrsaMinor','Sculptor','Sextans','HorologiumI','Reticulum_III','PhoenixII','UrsaMajorI','Carina','Hercules','Fornax','LeoIV','CanesVenaticiII','Columba_I','Indus_II','CanesVenaticiI','LeoII','LeoI','Eridanus_II']

Lat_lon = ['(220.48,50.43)','(315.38,-56.18)','(266.30,-49.74)','(152.46,37.44)','(158.58,56.78)','(241.89,83.61)','(313.29,-55.29)','(351.14,-51.94)','(328.04,-52.35)','(358.08,69.62)','(347.20,-42.10)','(86.37,34.72)','(104.97,44.80)','(287.53,-83.16)','(243.50,42.27)','(271.38,-54.74)','(273.88,-45.65)','(323.69,-59.74)','(159.43,54.41)','(260.11,-22.22)','(28.73,36.87)','(237.10,-65.65)','(265.44,56.51)','(113.58,82.70)','(231.62,-28.88)','(354.00,-37.40)','(74.31,79.82)','(220.17,67.23)','(225.99,49.11)','(249.78,-51.65)']

for i in range(1): 
    Lum, Flux, Name = dwarf_info(i)
    results[i,:] = np.percentile(Flux, [50,16,84])
    
    fore_flux = np.ones(1000)
    for q in range(len(fore_flux)):
        if fore_data[i,q]>0.0:
            fore_flux[q] = np.log10(fore_data[i,q]/2.984e-3)
        else:
            fore_flux[q] = -19.8
    
    fig = plt.figure(1,figsize=(6,4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121, gs[0])
    ax2 = fig.add_subplot(122, gs[1])
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.tick_right()
    ax2.tick_params(labelright='off')  # don't put tick labels at the top
    ax1.yaxis.tick_left()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((-d, +d), (-d, +d), **kwargs)  # bottom-right diagonal

    plt.gcf().subplots_adjust(bottom=0.13)
    #plt.figure(1)
    ax1.hist(Flux, bins=np.linspace(-20,-7,81), normed=1.0, facecolor='green',alpha=0.5)
    ax1.hist(fore_flux, bins=np.linspace(-20,-7,81), normed=1.0, facecolor='blue',alpha=0.5)
    ax2.hist(Flux, bins=np.linspace(-20,-7,81), normed=1.0, facecolor='green',alpha=0.5, label='dSph')
    ax2.hist(fore_flux, bins=np.linspace(-20,-7,81), normed=1.0, facecolor='blue',alpha=0.5, label='foreground')
    ax2.plot([],[],color='none',label='$<N_{\\rm MSP}>=%s$' % NMSP[i])
    ax2.plot([],[],color='none',label='(l,b)=%s' % Lat_lon[i])
    ax1.set_xlim(-20.5, -19.5)
    ax2.set_xlim(-17.0, -7.0)
    ax1.set_ylim(1e-5, 10)
    ax2.set_ylim(1e-5, 10)
    #plt.title('Flux PDF: %s' % Name, fontsize=18)
    ax.set_xlabel('Flux $>500$ MeV (ph cm$^{-2}$ s$^{-1}$)',labelpad=10)
    ax.set_ylabel('Counts (arb. units)',labelpad=10)
    #plt.xscale('log')
    ax2.set_yscale('log')
    ax1.set_yscale('log')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(reversed(handles), reversed(labels), loc=0, ncol=1, bbox_to_anchor=(0, 0, 1, 1), title='%s' % dSph_name[i],
           prop = fontP,fancybox=False,shadow=False,framealpha=0.0)
    plt.savefig('broken_plots/%s_flux.pdf' % Name)
    plt.show()

    print Name

