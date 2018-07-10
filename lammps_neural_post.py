# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:31:27 2018

@author: Nicholas
"""

from __future__ import division, print_function
import sys, pickle
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
property = 'radial_distribution'   # property for classification
n_press = 8                        # number of pressure datasets
n_dat = 48                         # number of temperature datasets
ntrainsets = 8                     # number of training sets
nsmpl = 1024                       # number of samples from each set
scaler = 'minmax'                  # data scaling method
network = 'keras_cnn1d'            # neural network type
reduc = False                      # reduction type
fit_func = 'logistic'              # fitting function

# element and pressure index choice
if '--element' in sys.argv:
    i = sys.argv.index('--element')
    el = sys.argv[i+1]
else:
    el = 'LJ'

# pressure
press = {'Ti': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
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
prefixes = ['%s.%s.%s.%d.lammps' % (name, el.lower(), lat[el], int(press[el][i])) for i in xrange(n_press)]
if reduc:
    network_pref = [network, property, scaler, reduc, fit_func, str(nsmpl)]
else:
    network_pref = [network, property, scaler, 'none', fit_func, str(nsmpl)]

mU = np.zeros((n_press, n_dat), dtype=float)
sU = np.zeros((n_press, n_dat), dtype=float)
mP = np.zeros((n_press, n_dat), dtype=float)
sP = np.zeros((n_press, n_dat), dtype=float)
mT = np.zeros((n_press, n_dat), dtype=float)
sT = np.zeros((n_press, n_dat), dtype=float)
msP = np.zeros((n_press, 2), dtype=float)
trans = np.zeros((n_press, 2), dtype=float)
for i in xrange(n_press):
    prefix = prefixes[i]
    outfile = '.'.join([prefix]+network_pref+['out'])
    # load simulation data
    N = pickle.load(open(prefix+'.natoms.pickle'))
    O = np.concatenate(tuple([j*np.ones(int(len(N)/n_dat), dtype=int) for j in xrange(n_dat)]), 0)
    # load potential data
    U = pickle.load(open(prefix+'.pe.pickle'))
    mU[i, :] = np.array([np.mean(U[O == j]) for j in xrange(n_dat)])
    sU[i, :] = np.array([np.std(U[O == j]) for j in xrange(n_dat)])
    # load pressure data
    P = pickle.load(open(prefix+'.virial.pickle'))
    mP[i, :] = np.array([np.mean(P[O == j]) for j in xrange(n_dat)])
    sP[i, :] = np.array([np.std(P[O == j]) for j in xrange(n_dat)])
    msP[i, :] = np.array([np.mean(P), np.std(P)], dtype=float)
    # load temperature data
    T = pickle.load(open(prefix+'.temp.pickle'))
    mT[i, :] = np.array([np.mean(T[O == j]) for j in xrange(n_dat)])
    sT[i, :] = np.array([np.std(T[O == j]) for j in xrange(n_dat)])
    with open(outfile, 'r') as fi:
        iters = iter(fi)
        for lina in iters:
            if 'transition | critical error' in lina:
                linb = iters.next()
                trans[i, :] = np.array(linb.strip().split()).astype(float)
    
base_pref = ['%s.%s.%s.lammps' % (name, el.lower(), lat[el])]+network_pref

cm = plt.get_cmap('plasma')
cscale = lambda i: (msP[i, 0]-np.min(mP))/np.max(mP)

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
for i in xrange(n_press):
    ax0.errorbar(mT[i], mP[i], xerr=sT[i], yerr=sP[i], color=cm(cscale(i)), label='P = %.1f' % press[el][i])
ax0.set_xlabel('T')
ax0.set_ylabel('P')
ax0.legend(loc='center right')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
for i in xrange(n_press):
    ax1.errorbar(mT[i], mU[i], xerr=sT[i], yerr=sU[i], color=cm(cscale(i)), label='P = %.1f' % press[el][i])
ax1.set_xlabel('T')
ax1.set_ylabel('U')
ax1.legend(loc='upper left')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.errorbar(trans[:, 0], msP[:, 0], xerr=trans[:, 1], yerr=msP[:, 1], color=cm(0.5))
ax2.set_xlabel('T')
ax2.set_ylabel('P')
ax2.legend(loc='lower right')

fig0.savefig('.'.join(base_pref+['pressure', 'png']))
fig1.savefig('.'.join(base_pref+['potential', 'png']))
fig2.savefig('.'.join(base_pref+['melting_curve', 'png']))