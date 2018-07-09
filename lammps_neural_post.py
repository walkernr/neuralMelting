# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:31:27 2018

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# plotting parameters
# plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
ftsz = 48
params = {'figure.figsize': (26, 20),
          'lines.linewidth': 4.0,
          'legend.fontsize': ftsz,
          'axes.labelsize': ftsz,
          'axes.titlesize': ftsz,
          'axes.linewidth': 2.0,
          'xtick.labelsize': ftsz,
          'xtick.major.size': 20,
          'xtick.major.width': 2.0,
          'ytick.labelsize': ftsz,
          'ytick.major.size': 20,
          'ytick.major.width': 2.0,
          'font.size': ftsz}
          # 'text.latex.preamble': r'\usepackage{amsmath}'r'\boldmath'}
plt.rcParams.update(params)

# simulation name
name = 'remcmc'
# run details
property = 'entropic_fingerprint'  # property for classification
n_press = 8                        # number of pressure datasets
n_dat = 48                         # number of temperature datasets
ntrainsets = 8                     # number of training sets
nsmpl = 1024                       # number of samples from each set
scaler = 'tanh'                    # data scaling method
network = 'keras_cnn1d'            # neural network type
reduc = 'pca'                      # reduction type
fit_func = 'logistic'              # fitting function

# element and pressure index choice
if '--element' in sys.argv:
    i = sys.argv.index('--element')
    el = sys.argv[i+1]
else:
    el = 'LJ'

# pressure
P = {'Ti': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
     'Al': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
     'Ni': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
     'Cu': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
     'LJ': np.linspace(1.0, 8.0, n_press, dtype=np.float64)}
# lattice type
lat = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}

# summary of input
print('------------------------------------------------------------')
print('input summary')
print('------------------------------------------------------------')
print('potential:                 %s' % el.lower()) 
print('number of pressures:       %d' % n_press)
print('number of temperatures:    %d' % n_dat)
print('number of samples:         %d' % nsmpl)
print('property:                  %s' % property)
print('training sets (per phase): %d' % ntrainsets)
print('scaler:                    %s' % scaler)
print('network:                   %s' % network)
print('fitting function:          %s' % fit_func)
print('reduction:                 %s' % reduc)
print('------------------------------------------------------------')

# file prefix
prefixes = ['%s.%s.%s.%d.lammps' % (name, el.lower(), lat[el], int(P[el][i])) for i in xrange(n_press)]
if reduc:
    network_pref = [network, property, scaler, reduc, fit_func]
else:
    network_pref = [network, property, scaler, 'none', fit_func]

mU = np.zeros((n_press, d_dat), dtype=float)
sU = np.zeros((n_press, d_dat), dtype=float)
mP = np.zeros((n_press, d_dat), dtype=float)
sP = np.zeros((n_press, d_dat), dtype=float)
mT = np.zeros((n_press, d_dat), dtype=float)
sT = np.zeros((n_press, d_dat), dtype=float)
trans = np.zeros((n_press, 2), dtype=float)
for i in xrange(len(P[el])):
    # prefix for output files
    in_pref = [prefixes[i]]+network_pref
    with open('.'.join(in_pref+[str(nsmpl), 'out']), 'w') as fi:
        lines = fi.readlines()
    mU[i, :] = np.array(lines[0].strip().split()).astype(float)
    sU[i, :] = np.array(lines[1].strip().split()).astype(float)
    mP[i, :] = np.array(lines[2].strip().split()).astype(float)
    sP[i, :] = np.array(lines[3].strip().split()).astype(float)
    mT[i, :] = np.array(lines[4].strip().split()).astype(float)
    sT[i, :] = np.array(lines[5].strip().split()).astype(float)
    trans[i, :] = np.array(lines[6].strip().split()).astype(float)
    
base_pref = ['%s.%s.%s.lammps' % (name, el.lower(), lat[el])]+network_pref

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
for i in xrange(len(prefixes)):
    ax0.errorbar(mT[i], mP[i], xerr=sT[i], yerr=sP[i], color=cm(0.1*i), label='P = %.1f' % P[el][i])
ax0.set_xlabel('T')
ax0.set_ylabel('P')
ax0.legend(loc='center right')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
for i in xrange(len(prefixes)):
    ax1.errorbar(mT[i], mU[i], xerr=sT[i], yerr=sU[i], color=cm(0.1*i), label='P = %.1f' % P[el][i])
ax1.set_xlabel('T')
ax1.set_ylabel('U')
ax1.legend(loc='upper left')

fig0.savefig('.'.join(base_pref+['press', 'png']))
fig1.savefig('.'.join(base_pref+['pot', 'png']))