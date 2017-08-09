import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import norm
from matplotlib.font_manager import FontProperties

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
fontP = FontProperties()
fontP.set_size(14)
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
            flux[j] = -30.0
    return lum, flux, name

results = np.zeros((30,3))
data = np.loadtxt('dSph_msp_list.txt')
NMSP = np.array(data) 

dSph_name = ['Segue 1','Tucana III','Reticulum II','Ursa Major II','Willman I','Coma Berenices','Tucana IV','Grus II','Tucana II','Bootes I','Indus I','Draco','Ursa Minor','Sculptor','Sextans','Horologium I','Reticulum III','Phoenix II','Ursa Major I','Carina','Hercules','Fornax','Leo IV','Canes Venatici II','Columba I','Indus II','Canes Venatici I','Leo II','Leo I','Eridanus II']

fore_data = np.loadtxt('flux_foreground.txt')
#Name = ['SegueI','Tucana_III','ReticulumII','UrsaMajorII','WillmanI','Coma_Berenices','Tucana_IV','Grus_II','TucanaII','BootesI','IndusI','Draco','UrsaMinor','Sculptor','Sextans','HorologiumI','Reticulum_III','PhoenixII','UrsaMajorI','Carina','Hercules','Fornax','LeoIV','CanesVenaticiII','Columba_I','Indus_II','CanesVenaticiI','LeoII','LeoI','Eridanus_II']

Lat_lon = ['(220.5,50.4)','(315.4,-56.2)','(266.3,-49.7)','(152.5,37.4)','(158.6,56.8)','(241.9,83.6)','(313.3,-55.3)','(351.1,-51.9)','(328.0,-52.4)','(358.1,69.6)','(347.2,-42.1)','(86.4,34.7)','(105.0,44.8)','(287.5,-83.2)','(243.5,42.3)','(271.4,-54.7)','(273.9,-45.7)','(323.7,-59.7)','(159.4,54.4)','(260.1,-22.2)','(28.7,36.9)','(237.1,-65.7)','(265.4,56.5)','(113.6,82.7)','(231.6,-28.9)','(354.0,-37.4)','(74.3,79.8)','(220.2,67.2)','(226.0,49.1)','(249.8,-51.7)']

for i in range(21,22): 
    Lum, Flux, Name = dwarf_info(i)
    results[i,:] = np.percentile(Flux, [50,16,84])
    
    fore_flux = np.ones(1000)
    for q in range(len(fore_flux)):
        if fore_data[i,q]>0.0:
            fore_flux[q] = np.log10(fore_data[i,q]/2.984e-3)
        else:
            fore_flux[q] = -30.0
       
    fig, ax = plt.subplots()
    plt.gcf().subplots_adjust(bottom=0.13)
    #Create Plot
    plt.figure(1)
    plt.hist(Flux, bins=np.linspace(-30,-7,121), normed=1.0, lw=2.0,histtype='step', cumulative=-1, edgecolor='green',alpha=0.5, label='dSph')
    plt.hist(fore_flux, bins=np.linspace(-30,-7,121), normed=1.0, lw=2.0,  histtype='step', cumulative=-1, edgecolor='blue',alpha=0.5, label='Foreground')
    plt.plot(1,1,color='none',label='$<N_{\\rm MSP}>=%s$' % NMSP[i])
    plt.plot(1,1,color='none',label='(l,b)=%s' % Lat_lon[i])
    plt.axvline(np.log10(1.1877362186922663e-10), color='k', lw=1.5, linestyle='dashed', label='LAT sensitivty')
    #plt.title('Flux PDF: %s' % Name, fontsize=18)
    plt.xlabel('Flux $>500$ MeV (ph cm$^{-2}$ s$^{-1}$)',labelpad=10)
    plt.ylabel('Complementary CDF',labelpad=10)
    #ax.text(-17.6, 0.05, 'Underflow Bins', fontsize=22, rotation='vertical')
    #plt.xscale('log')
    plt.yscale('log')
    plt.xlim([-16.0,-7.0])
    plt.ylim([1.0e-3,1.1])
    plt.xticks([-16.0,-15.0,-14.0,-13.0,-12.0,-11.0,-10.0,-9.0,-8.0,-7.0], ['$10^{-16}$', '', '$10^{-14}$','', '$10^{-12}$', '', '$10^{-10}$','', '$10^{-8}$', ''])
    handles, labels = ax.get_legend_handles_labels()
    #plt.legend(reversed(handles), reversed(labels), ncol=1, loc='lower left', bbox_to_anchor=(0,0,1,1), title=r'\underline{%s}' % dSph_name[i],prop = fontP,fancybox=False,shadow=False,framealpha=0.0)
    plt.legend(reversed(handles), reversed(labels), ncol=1, bbox_to_anchor=(.38,.67), title=r'\underline{%s}' % dSph_name[i],prop = fontP,fancybox=False,shadow=False,framealpha=0.0)
    plt.savefig('cum_dwarf_plots/%s_cum_flux.pdf' % Name)
    plt.show()
    
    print i, '  ', Name

