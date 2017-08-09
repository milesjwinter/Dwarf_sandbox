import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import integrate
import scipy.stats

############################################################
def LF(L, M=7.0E10, k = 19.55, Lb = 32.94, a1 = 1.45, a2 = 2.86):
    dN = 10.0**(-a1*L[L<=Lb]+k)
    dN = np.append(dN, 10.0**(-a2*(L[L>Lb]-Lb)-a1*Lb+k))
    return M*dN/7.0E10
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

Lum_vals = np.zeros(len(k))
for i in range(len(k)):
    dN = LF(logL, M=1.0, k = k[i], Lb = Lb[i], a1 = a1[i], a2 = a2[i])
    Lum_vals[i] = np.trapz(Lum*dN,Lum)


Log_Lum = np.log10(Lum_vals)
mean = np.percentile(Log_Lum,50)
lower = np.percentile(Log_Lum,16)
upper = np.percentile(Log_Lum,84)
print 10.0**mean
print 10.0**mean - 10.0**lower
print 10.0**upper - 10.0**mean
'''
#Create Plot
minorLocator = AutoMinorLocator()

bins = np.linspace(Log_Lum.min(),Log_Lum.max(),100)
fig, ax = plt.subplots()
plt.figure(1)
plt.hist(Log_Lum, bins=bins, normed=False, color='g', alpha=0.5)
#plt.title('LAT Detected MSPs',fontsize=18)
plt.xlabel('$\log_{10}(L_\gamma) (erg\cdot s^{-1})$',fontsize=18)
plt.ylabel('Counts',fontsize=18)
#ax.set_yscale('log')
#ax.set_xscale('log')
plt.axis((Log_Lum.min(),Log_Lum.max(),0.0,500.0))

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
