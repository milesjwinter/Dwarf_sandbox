import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import integrate
from scipy.interpolate import UnivariateSpline
import scipy.stats
from matplotlib.font_manager import FontProperties

############################################################
def LF(L, M=7.0E10, k = 19.55, Lb = 32.94, a1 = 1.45, a2 = 2.86):
    dN = 10.0**(-a1*L[L<=Lb]+k)
    dN = np.append(dN, 10.0**(-a2*(L[L>Lb]-Lb)-a1*Lb+k))
    return M*dN/7.0E10
    #return dN/7.0E10
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
fontP.set_size(16)
############################################################

#Pull in data from text file
data = np.loadtxt('data_points/fermi_mean.txt') #Load data
L = np.array(data[:,0]) # bin from Data
Ll = np.array(data[:,1]) # bin uncertanty from Data
Lu = np.array(data[:,2]) # bin uncertanty from Data
N = np.array(data[:,3]) # sources from Data
Nl = np.array(data[:,4]) # sources uncertanty from Data
Nu = np.array(data[:,5]) # sources uncertanty from Data

data = np.loadtxt('lf_stat_vals.txt') #Load data
dNl = np.array(data[:,0]) # bin from Data
dNh = np.array(data[:,1]) # bin from Data

data = np.loadtxt('hooper_lf.txt')
hooper_lum = np.array(data[:,0])
hooper_lf = UnivariateSpline(hooper_lum,data[:,1],s=0.0)

#Luminosity values
logL = np.linspace(31.0,36.0,2.0E4)
Lum = np.logspace(31.0,36.0,2.0E4)

#Fill luminosity function and calculate uncertainty
dN = LF(logL, M=7.0E10, k = 19.55, Lb = 32.94, a1 = 1.45, a2 = 2.86)
dN_int_e = np.trapz(Lum**2*dN,Lum)
hooper_int_e = np.trapz(hooper_lf(Lum),Lum)
Norm_e = dN_int_e/hooper_int_e

dN_int_m = np.trapz(Lum*dN,Lum)
hooper_int_m = np.trapz(hooper_lf(Lum)/Lum,Lum)
Norm_m = dN_int_m/hooper_int_m

#Fill luminosity functions 
dNF = LF(logL, M=7.0E10, k = 19.55, Lb = 32.94, a1 = 1.45, a2 = 2.86)
dNA = LF(logL, M=7.0E10, k = 19.50, Lb = 32.92, a1 = 1.45, a2 = 2.72)
dNFl = LF(logL, M=7.0E10, k = 19.17, Lb = 32.91, a1 = 1.45, a2 = 2.73)
dNAl = LF(logL, M=7.0E10, k = 19.45, Lb = 32.91, a1 = 1.46, a2 = 2.58)
dNFu = LF(logL, M=7.0E10, k = 19.65, Lb = 32.95, a1 = 1.44, a2 = 3.02)
dNAu = LF(logL, M=7.0E10, k = 19.57, Lb = 32.95, a1 = 1.45, a2 = 2.87)

Syst = np.array([dNA, dNF, dNFl, dNFu, dNAl, dNAu])
Systl = np.amin(Syst,axis=0)
Systh = np.amax(Syst,axis=0)
Sigma = np.std(Syst,axis=0)
'''
dNsyl = dN-Systl
dNstl = dN-dNl
dNsl = dN-np.sqrt(dNsyl**2+dNstl**2)
dNsyh = Systh-dN
dNsth = dNh-dN
dNsh = dN+np.sqrt(dNsyh**2+dNsth**2)
#write luminosity function to file
Lum_func = np.array([Lum, dN, dN-dNl, dNh-dN, Systl, Systh])

Lum_func = np.array([Lum, dN])
#np.savetxt('best_lum_func_new.txt',np.transpose(Lum_func))
'''

#Create Plots
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.13)
plt.figure(1)
#minorLocator = AutoMinorLocator()
plt.plot(Lum, Lum**2*dN, "r", linewidth=2.0, label='Best-fit LF (this study)')
#plt.plot(Lum, dNA, "b", linewidth=2.0, label='ATNF Mean')
#plt.plot(Lum, dNFu, "r", linewidth=2.0, ls='dashed', label='2PC Upper/Lower')
#plt.plot(Lum, dNFl, "r", linewidth=2.0, ls='dashed')
#plt.plot(Lum, dNAu, "b", linewidth=2.0, ls='dashed', label='ATNF Upper/Lower')
#plt.plot(Lum, dNAl, "b", linewidth=2.0, ls='dashed')
#plt.plot(Lum, Statl, "b", linewidth=2.0, ls='dashed')
#plt.plot(Lum, Stath, "r", linewidth=2.0, ls='dashed')
plt.fill_between(Lum, Lum**2*dNl, Lum**2*dNh , alpha=0.5, interpolate=True, facecolor='grey')
plt.plot(Lum, Lum**2*Systl, "k", ls='dashed', linewidth=1.0)
plt.plot(Lum, Lum**2*Systh, "k", ls='dashed', linewidth=1.0)
#plt.fill_between(Lum, Systl, Systh , alpha=0.4, interpolate=True, facecolor='grey', label='Syst. Unc.', hatch='//')
#plt.title('MSP Luminosity Function', fontsize=18)
plt.xlabel('$L_\gamma$ (erg/s)', labelpad=10)
plt.ylabel('$L_\gamma^2dN/dL_\gamma$ (erg/s)', labelpad=10)
plt.plot([], [], color='grey', linewidth=10, alpha=0.3, label='Statistical uncertainty')
plt.plot([], [], color='k', ls='dashed', linewidth=2.0, label='Systematic uncertainty')
plt.errorbar(L, L**2*N, xerr=[Ll,Lu], yerr=[L**2*Nl,L**2*Nu], fmt=',k', mfc='k', lw=2.0, mew=2.0, zorder=4, label='LAT MSPs: 2PC dist. w/ mean completeness')
#plt.plot(Lum,Norm_e*hooper_lf(Lum), "b", linewidth=2.0, ls='dashdot',label='Hooper $\&$ Mohlabeng (2016) LF: fixed lum.')
plt.plot(Lum,Norm_m*hooper_lf(Lum), "b", linewidth=2.0, ls='dashdot',label='Hooper $\&$ Mohlabeng (2016) LF')
plt.legend(loc=0, ncol=1, bbox_to_anchor=(0, 0, 1, 1),
           prop = fontP,fancybox=False,shadow=False,framealpha=0.0)
plt.yscale('log')
plt.xscale('log')
#plt.axis((31.5,35.5,0.0001,5))
#ax.xaxis.set_minor_locator(minorLocator)
#ax.yaxis.set_minor_locator(minorLocator)
#plt.tick_params(which='major', length=7, width=2)
#plt.tick_params(which='minor', length=4)
#plt.minorticks_on()
#plt.grid('off')
plt.show()

'''
fig, ax = plt.subplots()
plt.figure(1)
minorLocator = AutoMinorLocator()
#plt.errorbar(L, L**2*N, xerr=[Ll,Lu], yerr=[L**2*Nl,L**2*Nu], fmt='+', mfc='c', lw=1.0, mew=2.0, zorder=3)
plt.errorbar(L, N, xerr=[Ll,Lu], yerr=[Nl,Nu], fmt='+', mfc='c', lw=1.0, mew=2.0, zorder=3)
plt.plot(Lum, dN, "r", linewidth=2.0, zorder=2)
plt.fill_between(Lum, dNl, dNh , alpha=0.5, interpolate=True, facecolor='grey', zorder=1)
#plt.title('MSP Luminosity Function', fontsize=18)
plt.xlabel('$L_\gamma (erg\cdot s^{-1})$', fontsize=18)
plt.ylabel('$dN/dL_\gamma$', fontsize=18)
ax.set_yscale('log')
ax.set_xscale('log')
#plt.axis((31.5,35.5,0.0001,5))
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.show()
'''
