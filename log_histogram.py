import numpy as np
import pylab

#pylab.ion()

"""
# Helvetica font
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [
    r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
    r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
    r'\usepackage{helvet}',    # set the normal font here
    r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
    r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]  
"""

params = {
    #'backend': 'eps',
    'axes.labelsize': 18,
    #'text.fontsize': 12,           
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.major.size': 3,      # major tick size in points
    'xtick.minor.size': 1.5,    # minor tick size in points
    'xtick.major.size': 3,      # major tick size in points
    'xtick.minor.size': 1.5,    # minor tick size in points
    'text.usetex': True,
    #'axes.xmargin': 1.,
    #'figure.figsize': fig_size,
    'font.family':'serif',
    'font.serif':'Computer Modern Roman',
    'font.size': 14
    }
pylab.rcParams.update(params)

############################################################

def logHist(x, n_bins=50, **kwargs):
    log_x = np.log10(x)
    log_bins = np.linspace(np.percentile(log_x, 0.01), 
                           np.percentile(log_x, 99.99),
                           n_bins)
    delta_log_x = (log_bins[-1] - log_bins[0]) / (n_bins - 1.)
    

    weights = np.tile(1. / (delta_log_x * len(x)), len(x))
    results = pylab.hist(log_x, bins=log_bins, weights=weights, 
                         histtype='stepfilled', **kwargs)
    #print delta_log_x * np.sum(results[0])

############################################################

# Make some random data
n = 1000000
x_1 = 10**(0. + np.random.normal(size=n))
x_2 = 10**(3. + 0.1 * np.random.normal(size=n))
x_3 = 10**(6. + 0.03 * np.random.normal(size=n))

pylab.figure()
logHist(x_1, color='green')
logHist(x_2, color='blue')
logHist(x_3, color='red')

# Axis tricks
pylab.yscale('log')
pylab.xlim(-4, 7)
pylab.ylim(1.e-4, 1.e2)
if False:
    # Label ticks in linear space
    x_ticks = np.arange(-4, 8, 2)
    x_labels = []
    for x_tick in x_ticks:
        x_labels.append('$10^{{{0:d}}}$'.format(x_tick))
    pylab.xticks(x_ticks, x_labels)
    pylab.xlabel('Luminosity (Units)', labelpad=10) # Linear-space ticks
else:
    # Label ticks in log space
    pylab.xlabel('$\log_{10}$(Luminosity)') # Log-space ticks
pylab.ylabel('PDF', labelpad=10)
pylab.show()
############################################################
