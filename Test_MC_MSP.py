"""
Miles Winter
Monte Carlo incompleteness correction for MW MSPs
"""

from math import *
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator
import time

start_time = time.time()

#Pull in data from text file
data = np.loadtxt('lat_flux.txt') #Load data
latb = np.array(data[:,0]) # Lum from Data
FluxM = np.array(data[:,1]) # Num from Data
Flux = UnivariateSpline(latb, FluxM, s=0.0)

R0 = 15.0E3 #Radius of the Milky Way in pc
Z0 = 1.0E3 #Height of Milky Way in pc
pnts = 1000 #Base number of points (unscaled)
zpnts = pnts+1 #Number of points for Z function
rpnts = int(7.5*pnts)+1 # Number of points for R function
NMSP = 1E6 #Number of MC MSPs generated
NMC = 100
Lum_bins = np.logspace(31.75,35.25,8)
#Lum_bins = np.logspace(31.0,36.0,1E+3)

Hr = 1.0E3*np.random.lognormal(1.099,0.333,NMC) #Radial scale height in pc
Hz = 1.0E3*np.random.lognormal(-0.511,0.375,NMC) #Vertical scale height in pc

#Generate random z coordinates
Z_int_vals = np.random.uniform(0.0,Hz,size=(NMSP,NMC))
Z_sign = (-1.0)**np.random.randint(2,size=(NMSP,NMC))  
Z_mc = Z_sign*Hz*np.log(Hz/(Hz-Z_int_vals))
    
#Generate random r coordinates
R_int_vals = np.random.uniform(0.0,Hr,size=(NMSP,NMC))
R_mc = Hr*np.log(Hr/(Hr-R_int_vals))

#Generate random theta coordinates
Th_mc = np.random.uniform(0.0,2.0*np.pi,size=(NMSP,NMC))

#Convert MSP coordinates to cartesian and calculate b
X = R_mc*np.cos(Th_mc)
Y = R_mc*np.sin(Th_mc)
Z = Z_mc
D_mc = np.sqrt(X**2+Y**2+Z**2)
b_mc = np.arcsin(Z/D_mc)*180.0/np.pi

#Convert sun's coordinates to cartesian: (R,Th,Z)=(8.5kpc,0.0rad,20pc)
Rsun = 8.5E3 # in pc
Thsun = 0.0 # in radians
Xsun = Rsun*np.cos(Thsun) # in pc
Ysun = Rsun*np.sin(Thsun) # in pc
Zsun = 20.0 # in pc
    
#Calculate distance from MSPs to the sun (note dY=Y)
dX = X-Xsun
dZ = Z-Zsun
Dsq = dX**2+Y**2+dZ**2


Flux_th = np.zeros((NMC,NMSP))

for q in range(NMC):
    Flux_th[q,:] = Flux(b_mc[:,q])
    print q

Lum_th = np.transpose(Flux_th)*Dsq*1.19651E38
Lum_th = Lum_th.reshape((NMSP,NMC,1))

#Calculate completeness fraction
Ndet = np.sum(Lum_th<=Lum_bins,axis=0,dtype='float')
completeness = Ndet/float(NMSP)

'''
completeness = np.zeros((NMC,len(Lum_bins)))

for q in range(NMC):
    #Define distance threshold for each MSP luminosity bin
    Flux_th = Flux(b_mc[:,q])
    Lum_th = Flux_th*Dsq[:,q]*1.19651E38
    Lum_th = Lum_th.reshape((NMSP,1))

    #Calculate completeness fraction
    Ndet = np.sum(Lum_th<=Lum_bins,axis=0,dtype='float')
    completeness[q,:] = Ndet/float(NMSP)
    print q
'''
#calculate mean/std and store values
mean = np.mean(completeness,axis=0)
sigma = np.std(completeness,axis=0)
comp_unc = np.array([Lum_bins, mean, sigma])
#np.savetxt('comp_unc.txt',np.transpose(comp_unc))

job_time = time.time() - start_time

print ""
print "Number of Simulated MSPs = ", NMSP, "   |   Time = ",job_time
print "---------------------------------"
for q in range(len(Lum_bins)):
    print "Bin ",q
    print "    Luminosity = ", Lum_bins[q], 
    print "    Completeness = ",mean[q], ' +- ',sigma[q]
    print ""

'''
#Create completness plot
fig, ax = plt.subplots()
plt.figure(1)
minorLocator = AutoMinorLocator()
plt.plot(Lum_bins,100.0*mean, 'r', linewidth=2.0)
ax.fill_between(Lum_bins, 100.0*(mean-sigma), 100.0*(mean+sigma), facecolor='grey', alpha=0.5, interpolate=True)
plt.title('Estimated Completeness vs $\gamma$-Ray Luminosity',fontsize=18)
plt.xlabel('$\log_{10}{L_\gamma} (erg\cdot s^{-1})$',fontsize=15)
plt.ylabel('Completeness ($\%$)',fontsize=15)

#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
ax.set_xscale('log')
ax.set_yscale('log')
plt.grid('off')
plt.show()
'''
'''
#Generate plot of MC MSPs
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z)
ax.scatter(Xsun, Ysun, Zsun, s=80 ,c="r")

# Cylinder
xc=np.linspace(-R0, R0, 100)
zc=np.linspace(-Z0, Z0, 100)
Xc, Zc=np.meshgrid(xc, zc)
Yc = np.sqrt(R0**2-Xc**2)

# Draw parameters
rstride = 20
cstride = 10
ax.plot_surface(Xc, Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
ax.plot_surface(Xc, -Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)

ax.set_xlabel('X(pc)')
ax.set_ylabel('Y(pc)')
ax.set_zlabel('Z(pc)')
plt.show()

plt.figure(2)
plt1 = plt.subplot(311)
plt.hist(R_mc, bins = 200, histtype='stepfilled', normed=False, color='b', alpha=0.5)
plt.xlabel('r')
plt.grid('on')
plt2 = plt.subplot(312)
plt.hist(Z_mc, bins = 80, histtype='stepfilled', normed=False, color='r', alpha=0.5)
plt.xlabel('z')
plt.grid('on')
plt3 = plt.subplot(313)
plt.hist(theta_mc, bins = 50, histtype='stepfilled', normed=False, color='g', alpha=0.5)
plt.xlabel('$\\theta$')
plt.grid('on')
plt.show()
'''
