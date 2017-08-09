import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import integrate
import scipy.stats

############################################################

def f(L, M=7.0E10, k = 1.686E+15/7.0E10, Lb = 5.603E+32, a1 = 1.352, a2 = 2.348):
    """
    Broken power law based on fit to fermi data
    """
    dN = 0.0
    if (L <= Lb):
        dN = L*(k*M)*(L)**(-a1)

    else:
        dN = L*(k*M)*(Lb)**(a2-a1)*(L)**(-a2)

    return dN

############################################################

# Load completeness fraction 
data = np.loadtxt('comp_unc.txt') #Load data
Lum = np.array(data[:,0]) #Luminosity bins
Mean = np.array(data[:,1]) #Mean completeness
Sigma = np.array(data[:,2]) #Unc in completeness

#Vectorize functions 
vecf = np.vectorize(f)

#Fill luminosity function and calculate uncertainty
dN = vecf(Lum, M=1.0)
#dNn = vecf(Lum, M=1.0, k = (1.0E+15)/7.0E10, Lb = 5.603E+32-3.194E+30, a1 = 1.352+0.088, a2 = 2.348+0.231)
#dNp = vecf(Lum, M=1.0, k = (1.0E+16)/7.0E10, Lb = 5.603E+32+3.194E+30, a1 = 1.352-0.088, a2 = 2.348-0.231)

Nn = dN*(1.0-np.sqrt((Sigma/Mean)**2+1.0/(Mean*dN)))
Np = dN*(1.0+np.sqrt((Sigma/Mean)**2+1.0/(Mean*dN)))

np.savetxt('lum_func.txt',dN)

'''
#Create Plot
minorLocator = AutoMinorLocator()

fig, ax = plt.subplots()
plt.figure(1)
plt.plot(Lum, dN, "r", linewidth=2.0)
plt.fill_between(Lum, Nn, Np, where=(Nn<Np),alpha=0.5,interpolate=True, facecolor='grey')
#plt.title('MSP Luminosity Function')
plt.xlabel('$L_\gamma (erg\cdot s^{-1})$')
plt.ylabel('$dN/dL_\gamma$')
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
