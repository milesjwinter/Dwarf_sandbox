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

# data from text file
data = np.loadtxt('j_factor_kin_UL.txt') #Load data
Jk = np.array(data[:,0]) #J-factor kinematic
Jku = np.array(data[:,1]) #J-factor kinematic uncertainty 
Fk = np.array(data[:,2]) #Flux estimate
Fkl = np.array(data[:,3]) #Flux lower uncertainty
Fkh = np.array(data[:,4]) #Flux upper uncertainty

# data from text file
data = np.loadtxt('j_factor_est_UL.txt') #Load data
Je = np.array(data[:,0]) #J-factor estimated
Jeu = np.array(data[:,1]) #J-factor estimated uncertainty
Fe = np.array(data[:,2]) #Flux estimate
Fel = np.array(data[:,3]) #Flux lower uncertainty
Feh = np.array(data[:,4]) #Flux upper uncertainty

#Flux Upper Limit percentiles
Jrange = np.linspace(17.0,20.0,101)
F16 = 6.270410884424026e-11*np.ones(len(Jrange))
F50 = 1.1877362186922663e-10*np.ones(len(Jrange))
F84 = 2.2939683867012122e-10*np.ones(len(Jrange))

#DM model
DM = 3.0e-26*8.02/(8.0*np.pi*25.0**2)*np.logspace(17.0,20.0,101)


#Create Plot
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.13)
plt.figure(1)
plt.yscale('log')
plt.xlabel('J-Factor (GeV$^2/$cm$^{5}$)', labelpad=10)
plt.ylabel('Flux $>500$ MeV (ph cm$^{-2}$ s$^{-1}$)', labelpad=10)
plt.plot(Jrange,F50,'k',lw=1.0, ls='dashed')
plt.plot(Jrange, DM, 'r', lw=1.0, label='$\chi \chi \\rightarrow b \\bar{b}$ ; m = 25 GeV ; $\langle \sigma v \\rangle = 3 \\times 10^{-26}$ cm$^3$ s$^{-1}$')
plt.fill_between(Jrange, F16, F84, facecolor='grey', alpha=.4)
plt.errorbar(Jk, Fk, xerr=Jku, yerr=[Fkl,Fkh], fmt='ob', mew=1.0, capsize=4,  mec='b', label='$\\langle$Flux$\\rangle$ with $3\sigma$ UL; dSphs with measured J-factors')
plt.errorbar(Je, Fe, yerr=[Fel,Feh], fmt='ob', mew=1.0, capsize=4, mfc='white', mec='b', label='$\\langle$Flux$\\rangle$ with $3\sigma$ UL; candidate dSphs with predicted J-factors')
#plt.errorbar([19.65,19.5,19.35],[3.3e-16,3.3e-16,3.3e-16],yerr=[[3.15e-16,2.65e-16,2.3e-16],[5.72e-16,5.46e-16,5.30e-16]],fmt='ok', capsize=4,mfc='k',mec='k')
plt.plot([], [], color='grey', linewidth=10, alpha=0.4, label='Typical limit from high latitude blank fields')
plt.errorbar([19.85],[1.0e-12],yerr=[[6.58e-13],[1.59e-12]],fmt='ok', capsize=4,mfc='k',mec='k',label='$\\langle$Flux$\\rangle$ uncertainty interval; equal for all dSphs')
plt.text(18.22,7.3E-12, 'For',fontsize=12)
plt.text(19.52,4.0e-15, 'Seg 1',fontsize=12)
plt.text(18.82,2.0E-13, 'Dra',fontsize=12)
plt.text(18.85, 6.0e-13,'UMi',fontsize=12)
plt.text(18.64, 3.5e-12,'Scl',fontsize=12)
plt.text(19.34, 1.30e-14,'Ret II',fontsize=12)
plt.text(17.42, 6.4e-16,'Ind II',fontsize=12)
plt.text(17.73, 4.9e-13,'Leo I',fontsize=12)
plt.text(18.43, 5.77e-13,'Sex',fontsize=12)
#plt.text(19.85,4.2e-11,'$\delta M_*=.2$ dex',fontsize=12,rotation='vertical')
plt.ylim(1.0e-16,1.0e-8)
plt.xticks([17.0,18.0,19.0,20.0], ['$10^{17}$', '$10^{18}$', '$10^{19}$', '$10^{20}$'])
plt.legend(ncol=1, bbox_to_anchor=(0.71,1.02),
           prop = fontP, framealpha=0.0, fancybox=False, shadow=False)
plt.show()
