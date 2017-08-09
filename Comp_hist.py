import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('comp_array_5000.txt')
Comp7 = data[:,0]
plt.figure(1)
#count, bins, ignored = plt.hist(Comp7, 50, normed=True, align='mid')
x, bins,p=plt.hist(Comp7, 50, normed=1, align='mid')
#plt.axis('tight')
plt.show()
