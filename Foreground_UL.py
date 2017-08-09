import numpy as np

fore_data = np.loadtxt('flux_foreground.txt')
fore_UL = np.percentile(fore_data,[84.,97.5,99.85],axis=1)

np.savetxt('fore_UL.txt',np.transpose(fore_UL))
