import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import integrate
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
data = np.loadtxt('fermi_atnf_msp.txt') #Load data
LumF = np.array(data[:,0]) #Fermi Luminosities 
LumA = np.array(data[:,1]) #ATNF luminosities

#Pull in data from text file
data = np.loadtxt('fermi_bins.txt') #Load data
Bf = np.array(data[:,0]) # bin from Data
Bfn = np.array(data[:,1]) # bin uncertanty from Data
Bfp = np.array(data[:,2]) # bin uncertanty from Data
Sf = np.array(data[:,3]) # sources from Data
Sfu = np.array(data[:,4]) # sources uncertanty from Data

#Pull in data from text file
data = np.loadtxt('atnf_bins.txt') #Load data
Ba = np.array(data[:,0]) # bin from Data
Ban = np.array(data[:,1]) # bin uncertanty from Data
Bap = np.array(data[:,2]) # bin uncertanty from Data
Sa = np.array(data[:,3]) # sources from Data
Sau = np.array(data[:,4]) # sources uncertanty from Data

#define Fermi broken Power Law Fit line
def f(L, M=7.0E10, k = 2.306E+18/7.0E10, Lb = 9.659E+32, a1 = 1.451, a2 = 2.417):
    dN = 0.0
    
    if (L <= Lb):
        dN = (k*M)*(L)**(-a1)

    else:
        dN = (k*M)*(Lb)**(a2-a1)*(L)**(-a2)   

    return dN

#define Fermi Single Power Law Fit line
def fsingle(L, M=7.0E10, k = 44.75, a = 2.258):
    dN = M*10.0**(-a*np.log10(L)+k)/7.0E10
    return dN

#define ATNF Broken Power Law Fit line
def a(L, M=7.0E10, k = 3.248E+9/7.0E10, Lb = 4.78E+32, a1 = 1.17, a2 = 2.327):
    dN = 0.0

    if (L <= Lb):
        dN = L**2*(k*M)*(L)**(-a1)

    else:
        dN = L**2*(k*M)*(Lb)**(a2-a1)*(L)**(-a2)

    return dN

#define ATNF Single Power Law Fit line
def asingle(L, M=7.0E10, k = 41.08, a = 2.148):
    dN = M*10.0**(-a*np.log10(L)+k)/7.0E10
    return dN

points = np.logspace(31.0,36.0,1E4)
vecf = np.vectorize(f)
dNfM = vecf(points, M=7.0E10)
dNfH = vecf(points, M=7.0E10, k = (2.306E+18+6.183E18)/7.0E10, Lb = 9.659E+32+1.656E33, a1 = 1.451-0.037, a2 = 2.417-0.400)
dNfL = vecf(points, M=7.0E10, k = (2.306E+18+6.183E18)/7.0E10, Lb = 1.0E+32, a1 = 1.451+0.037, a2 = 2.417+0.400)
#dNfn  = dNf-np.sqrt(dNf/points)
#dNfp  = dNf+np.sqrt(dNf/points)
#veca = np.vectorize(a)
#dNa = veca(points,M=1.0)
mw = 7.0E+10

print 'mean ', np.trapz(dNfM, points)
print 'High ', np.trapz(dNfH,points)
print 'Low ', np.trapz(dNfL,points)

#Create Plot
#minorLocator = AutoMinorLocator()

#bins = 10.0**np.linspace(31.5,35.5,9)
bins = np.logspace(31.5,35.5,9)

fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.13)
plt.figure(1)
plt.hist(LumF, bins=bins, histtype='step', normed=False, color='r', alpha=1.0, linewidth=2.0, label='2PC Distances')
plt.hist(LumA, bins=bins, histtype='step', normed=False, color='b', alpha=1.0, linestyle=('dashed'), linewidth=2.0, label='ATNF Distances')
#plt.hist(LumA, bins=bins, histtype='step', normed=False, color='b', alpha=1.0, linestyle=('dashed'), linewidth=1.0, label='ATNF Incomplete')
#plt.errorbar(Bf, Bf**2*Sf/mw, xerr=[Bfn,Bfp], yerr=Bf**2*Sfu/mw, fmt='or', linewidth=2.0, label='LAT Distances')
#plt.errorbar(Ba, Ba**2*Sa/mw, xerr=[Ban,Bap], yerr=Ba**2*Sau/mw, fmt='^b', linewidth=2.0, label='ATNF Distances')
#plt.plot(points, dNfM, "r", linewidth=2.0)
#plt.fill_between(points, dNfL, dNfH, where=(dNfL<dNfH),alpha=0.5,interpolate=True, facecolor='grey')
#plt.plot(points, dNa, "b", linewidth=2.0)
#plt.title('LAT Detected MSPs',fontsize=18)
plt.xlabel('$L_\gamma$ (erg/s)', labelpad=10)
plt.ylabel('Counts', labelpad=10)
plt.legend(ncol=1, loc=0, bbox_to_anchor=(0,0,1,1), framealpha=0.0, prop=fontP, fancybox=False, shadow=False)
#ax.set_yscale('log')
plt.xscale('log')
#plt.axis((1.0E31,1.0E36,0.0,20))

#Format tick marks
#ax.xaxis.set_minor_locator(minorLocator)
#ax.yaxis.set_minor_locator(minorLocator)
#plt.tick_params(which='major', length=7, width=2)
#plt.tick_params(which='minor', length=4)
#plt.minorticks_on()
#plt.grid('off')

plt.show()
'''
#Create Plot
minorLocator = AutoMinorLocator()

fig, ax = plt.subplots()
plt.figure(2)
plt.plot(points, dNf, "r", linewidth=2.0)
#plt.fill_between(points, dNfn, dNfp, where=(dNfn<dNfp))
#plt.plot(points, dNa, "b", linewidth=2.0)
#plt.title('MSP Luminosity Function')
plt.xlabel('$L_\gamma (erg\cdot s^{-1})$')
plt.ylabel('$dN/dL_\gamma$ ')
#plt.legend(loc="upper right")
ax.set_yscale('log')
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
