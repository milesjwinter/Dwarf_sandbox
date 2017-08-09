import numpy as np

#Function to read in data
def dwarf_info(i):
    dwarf = ['SegueI','Tucana_III','ReticulumII','UrsaMajorII','WillmanI','ComaBerenices','Tucana_IV','Grus_II','TucanaII','BootesI','IndusI','Draco','UrsaMinor','Sculptor','Sextans','HorologiumI','Reticulum_III','PhoenixII','UrsaMajorI','Carina','Hercules','Fornax','LeoIV','CanesVenaticiII','Columba_I','Indus_II','CanesVenaticiI','LeoII','LeoI','EridanusII']
    Dist = np.array([23.0,25.0,32.0,32.0,38.0,44.0,48.0,53.0,58.0,66.0,69.0,76.0,76.0,86.0,86.0,87.0,92.0,95.0,97.0,105.0,132.0,147.0,154.0,160.0,182.0,214.0,218.0,233.0,254.0,330.0])
    name = dwarf[i]
    label = 'dwarf_text/%s.txt' % name
    lum = np.loadtxt(label)
    flux = lum/(4.0*np.pi*(Dist[i]*3.0857E+21)**2*2.984e-3)
    return np.percentile(flux,[84.,97.5,99.85])

Flux_UL = np.zeros((30,3))
for i in range(30):
    Flux_UL[i,:] = dwarf_info(i)

np.savetxt('pois_UL.txt',Flux_UL)
