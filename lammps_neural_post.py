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
# element
if '--element' in sys.argv:
    i = sys.argv.index('--element')
    el = sys.argv[i+1]
else:
    el = 'LJ'
# number of pressure datasets
if '--npress' in sys.argv:
    i = sys.argv.index('--npress')
    npress = int(sys.argv[i+1])
else:
    npress = 8
# pressure range
if '--rpress' in sys.argv:
    i = sys.argv.index('--rpress')
    lpress = float(sys.argv[i+1])
    hpress = float(sys.argv[i+2])
else:
    lpress = 1.0
    hpress = 8.0
# number of temperature datasets
if '--ntemp' in sys.argv:
    i = sys.argv.index('--ntemp')
    ntemp  = int(sys.argv[i+1])
else:
    ntemp = 48
# property for classification
if '--property' in sys.argv:
    i = sys.argv.index('--property')
    property = sys.argv[i+1]
else:
    property = 'entropic_fingerprint'
# number of samples from each set
if '--nsmpl' in sys.argv:
    i = sys.argv.index('--nsmpl')
    nsmpl = int(sys.argv[i+1])
else:
    nsmpl = 1024
# number of training sets
if '--ntrain' in sys.argv:
    i = sys.argv.index('--ntrain')
    ntrain = int(sys.argv[i+1])
else:
    ntrain = 8
# data scaling method
if '--scaler' in sys.argv:
    i = sys.argv.index('--scaler')
    scaler = sys.argv[i+1]
else:
    scaler = 'tanh'
# reduction type
if '--reduction' in sys.argv:
    i = sys.argv.index('--reduction')
    reduc = sys.argv[i+1]
else:
    reduc = 'pca'
# neural network type
if '--network' in sys.argv:
    i = sys.argv.index('--network')
    network = sys.argv[i+1]
else:
    network = 'keras_cnn1d'
# fitting function
if '--fitfunc' in sys.argv:
    i = sys.argv.index('--fitfunc')
    fitfunc = sys.argv[i+1]
else:
    fitfunc = 'logistic'


# pressure
press = np.linspace(lpress, hpress, npress, dtype=np.float64)
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
print('number of pressures:       %d' % npress)  
print('number of temps:           %d' % ntemp)
print('property:                  %s' % property)
print('number of samples:         %d' % nsmpl)
print('training sets (per phase): %d' % ntrain)
print('scaler:                    %s' % scaler)
print('reduction:                 %s' % reduc)
print('network:                   %s' % network)
print('fitting function:          %s' % fitfunc)
print('------------------------------------------------------------')

# file prefix
prefixes = ['%s.%s.%s.%d.lammps' % (name, el.lower(), lat[el], int(press[i])) for i in xrange(npress)]
if reduc:
    network_pref = [network, property, scaler, reduc, fit_func, str(nsmpl)]
else:
    network_pref = [network, property, scaler, 'none', fit_func, str(nsmpl)]

mU = np.zeros((npress, ntemp), dtype=float)
sU = np.zeros((npress, ntemp), dtype=float)
mP = np.zeros((npress, ntemp), dtype=float)
sP = np.zeros((npress, ntemp), dtype=float)
mT = np.zeros((npress, ntemp), dtype=float)
sT = np.zeros((npress, ntemp), dtype=float)
msP = np.zeros((npress, 2), dtype=float)
trans = np.zeros((npress, 2), dtype=float)
print('pressure', 'temperature')
print('------------------------------------------------------------')
for i in xrange(npress):
    prefix = prefixes[i]
    outfile = '.'.join([prefix]+network_pref+['out'])
    # load simulation data
    N = pickle.load(open(prefix+'.natoms.pickle'))
    O = np.concatenate(tuple([j*np.ones(int(len(N)/ntemp), dtype=int) for j in xrange(ntemp)]), 0)
    # load potential data
    U = pickle.load(open(prefix+'.pe.pickle'))
    mU[i, :] = np.array([np.mean(U[O == j]) for j in xrange(ntemp)])
    sU[i, :] = np.array([np.std(U[O == j]) for j in xrange(ntemp)])
    # load pressure data
    P = pickle.load(open(prefix+'.virial.pickle'))
    mP[i, :] = np.array([np.mean(P[O == j]) for j in xrange(ntemp)])
    sP[i, :] = np.array([np.std(P[O == j]) for j in xrange(ntemp)])
    msP[i, :] = np.array([np.mean(P), np.std(P)], dtype=float)
    # load temperature data
    T = pickle.load(open(prefix+'.temp.pickle'))
    mT[i, :] = np.array([np.mean(T[O == j]) for j in xrange(ntemp)])
    sT[i, :] = np.array([np.std(T[O == j]) for j in xrange(ntemp)])
    with open(outfile, 'r') as fi:
        iters = iter(fi)
        for lina in iters:
            if 'transition | critical error' in lina:
                linb = iters.next()
                trans[i, :] = np.array(linb.strip().split()).astype(float)
    print('%.2f %.2f' % (msP[i, 0], trans[i, 0]))
print('------------------------------------------------------------')
    
base_pref = ['%s.%s.%s.lammps' % (name, el.lower(), lat[el])]+network_pref

cm = plt.get_cmap('plasma')
cscale = lambda i: (msP[i, 0]-np.min(mP))/np.max(mP)

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
for i in xrange(npress):
    ax0.errorbar(mT[i], mP[i], xerr=sT[i], yerr=sP[i], color=cm(cscale(i)), label='P = %.1f' % press[el][i])
    ax0.axvline(trans[i, 0], color=cm(cscale(i)))
ax0.set_xlabel('T')
ax0.set_ylabel('P')
ax0.legend(loc='center right')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
for i in xrange(npress):
    ax1.errorbar(mT[i], mU[i], xerr=sT[i], yerr=sU[i], color=cm(cscale(i)), label='P = %.1f' % press[el][i])
    ax1.axvline(trans[i, 0], color=cm(cscale(i)))
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