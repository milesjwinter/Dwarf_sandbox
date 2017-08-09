import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import integrate
import scipy.stats

############################################################
def LF(L, M=7.0E10, k = 19.55, Lb = 32.94, a1 = 1.45, a2 = 2.86):
    dN = -a1*L[L<=Lb]+k
    dN = np.append(dN, -a2*(L[L>Lb]-Lb)-a1*Lb+k)
    return dN
############################################################

#Pull in data from text file
data = np.loadtxt('trace_array.txt') #Load data
k = np.array(data[0,:]) 
Lb = np.array(data[1,:]) 
a1 = np.array(data[2,:]) 
a2 = np.array(data[3,:])

#Luminosity values
logL = np.linspace(31.0,36.0,2.0E4)
Lum = np.logspace(31.0,36.0,2.0E4)

dNm = LF(logL, M=7.0E10, k = 19.55, Lb = 32.94, a1 = 1.45, a2 = 2.86)
dN = np.zeros((len(logL),len(k)))
for q in range(len(k)):
    dN[:,q] = LF(L=logL, M=7.0E10, k=k[q], Lb=Lb[q], a1=a1[q], a2=a2[q])

#dNs = np.std(dN, axis=1) 
#dNl = 10.0**(dNm-dNs)
#dNh = 10.0**(dNm+dNs)
#stat_array = np.array([dNl,dNh])
stat_array = 10.0**np.percentile(dN,[16,84],axis=1)
np.savetxt('lf_stat_vals.txt', np.transpose(stat_array))

'''
#Create Plot
minorLocator = AutoMinorLocator()
bins = np.logspace(np.log10(dN[19000,:].min()),np.log10(dN[19000,:].max()),100)
fig, ax = plt.subplots()
plt.figure(1)
plt.hist(dN[19000,:], bins=bins, normed=False, color='g', alpha=0.5)
#plt.title('LAT Detected MSPs',fontsize=18)
plt.xlabel('$\log_{10}(L_\gamma) (erg\cdot s^{-1})$',fontsize=18)
plt.ylabel('Counts',fontsize=18)
#ax.set_yscale('log')
ax.set_xscale('log')
#plt.axis((Log_Lum.min(),Log_Lum.max(),0.0,500.0))
#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
#plt.xlim([Log_Lum.min(),Log_Lum.max()])
plt.grid('off')
plt.show() 
'''
'''
#Create Plots
fig, ax = plt.subplots()
plt.figure(1)
minorLocator = AutoMinorLocator()
plt.plot(logL, dNm, "r", linewidth=2.0)
plt.fill_between(logL, dNm-dNs, dNm+dNs , alpha=0.3, interpolate=True, facecolor='grey')
plt.xlabel('$L_\gamma (erg\cdot s^{-1})$', fontsize=18)
plt.ylabel('$dN/dL_\gamma$', fontsize=18)
#plt.legend(loc="upper right")
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.axis((31.5,35.5,0.0001,5))
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.show()
'''
