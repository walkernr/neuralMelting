# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:31:27 2018

@author: Nicholas
"""

from __future__ import division, print_function
import sys, pickle
import numpy as np
from scipy.stats import linregress
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
if '--name' in sys.argv:
    i = sys.argv.index('--name')
    name = sys.argv[i+1]
else:
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
# include tsne results
if '--tsne' in sys.argv:
    tsne = True
else:
    tsne = False


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
print('neural network summary')
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

if tsne:
    tsneproperty = 'entropic_fingerprint'  # property for classification
    tsnensmpl = 512                        # number of samples per dataset
    tsnescaler = 'tanh'                    # data scaling method
    tsnereduc = 'tsne'                     # reduction method
    tsneclust = 'spectral'                 # clustering method

# file prefix
prefixes = ['%s.%s.%s.%d.lammps' % (name, el.lower(), lat[el], int(press[i])) for i in xrange(npress)]
if reduc:
    neurpref = [network, property, scaler, reduc, fitfunc, str(nsmpl)]
else:
    neurpref = [network, property, scaler, 'none', fitfunc, str(nsmpl)]

mU = np.zeros((npress, ntemp), dtype=float)
sU = np.zeros((npress, ntemp), dtype=float)
mP = np.zeros((npress, ntemp), dtype=float)
sP = np.zeros((npress, ntemp), dtype=float)
mT = np.zeros((npress, ntemp), dtype=float)
sT = np.zeros((npress, ntemp), dtype=float)
msP = np.zeros((npress, 2), dtype=float)
neurtrans = np.zeros((npress, 2), dtype=float)
print('neural network')
print('pressure', 'temperature')
print('------------------------------------------------------------')
for i in xrange(npress):
    prefix = prefixes[i]
    neurfile = '.'.join([prefix]+neurpref+['out'])
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
    with open(neurfile, 'rb') as fi:
        iters = iter(fi)
        for lina in iters:
            if 'transition | critical error' in lina:
                linb = iters.next()
                neurtrans[i, :] = np.array(linb.strip().split()).astype(float)
    print('%.2f %.2f' % (msP[i, 0], neurtrans[i, 0]))
print('------------------------------------------------------------')

if tsne:
    tsnepref = [tsneproperty, tsnescaler, tsnereduc, tsneclust, str(tsnensmpl)]
    tsnetrans = np.zeros((npress, 2), dtype=float)
    print('t-sne')
    print('pressure', 'temperature')
    print('------------------------------------------------------------')
    for i in xrange(npress):
        prefix = prefixes[i]
        tsnefile = '.'.join([prefix]+tsnepref+['out'])
        with open(tsnefile, 'rb') as fi:
            iters = iter(fi)
            for lina in iters:
                if 'transition | critical error' in lina:
                    linb = iters.next()
                    tsnetrans[i, :] = np.array(linb.strip().split()).astype(float)
        print('%.2f %.2f' % (msP[i, 0], tsnetrans[i, 0]))
    print('------------------------------------------------------------')
    
base_pref = ['%s.%s.%s.lammps' % (name, el.lower(), lat[el])]+neurpref

cm = plt.get_cmap('plasma')
cscale = lambda i: (msP[i, 0]-np.min(mP))/np.max(mP)

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
for i in xrange(npress):
    ax0.errorbar(mT[i], mP[i], xerr=sT[i], yerr=sP[i], color=cm(cscale(i)), alpha=0.5, label=r'$P = %.1f$' % press[i])
    ax0.axvline(neurtrans[i, 0], color=cm(cscale(i)))
ax0.set_xlabel(r'$T$')
ax0.set_ylabel(r'$P$')
ax0.legend(loc='center right')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
for i in xrange(npress):
    ax1.errorbar(mT[i], mU[i], xerr=sT[i], yerr=sU[i], color=cm(cscale(i)), alpha=0.5, label='P = %.1f' % press[i])
    ax1.axvline(neurtrans[i, 0], color=cm(cscale(i)))
ax1.set_xlabel(r'$T$')
ax1.set_ylabel(r'$U$')
ax1.legend(loc='upper left')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
if el == 'LJ':
    # ref 1: http://paros.princeton.edu/cbe422/MP.pdf
    # ref 2: http://dirac.ruc.dk/~urp/interface.pdf
    litpress0 = np.array([1, 5, 10])
    litpress1 = np.array([0.928, 2.185, 3.514, 4.939, 7.921])
    littemp0 = np.array([0.77, 1.061, 1.379])
    littemp1 = np.array([0.7, 0.8, 0.9, 1.0, 1.2])
    litslp0, litint0 = linregress(littemp0, litpress0)[:2]
    litslp1, litint1 = linregress(littemp1, litpress1)[:2]
    ax2.scatter(littemp0, litpress0, color=cm(0.25), s=240, edgecolors='none', marker='*', label=r'$\mathrm{Literature\enspace (full\enspace potential)}$')
    ax2.plot(littemp0, litslp0*littemp0+litint0, color=cm(0.25))
    ax2.scatter(littemp1, litpress1, color=cm(0.375), s=240, edgecolors='none', marker='*', label=r'$\mathrm{Literature\enspace} (r_c = 2.5)$')
    ax2.plot(littemp1, litslp1*littemp1+litint1, color=cm(0.375))
ax2.errorbar(neurtrans[:, 0], msP[:, 0], xerr=neurtrans[:, 1], yerr=msP[:, 1], color=cm(0.5), fmt='o', label=r'$\mathrm{Keras\enspace CNN-1D}$')
neurslp, neurint = linregress(neurtrans[:, 0], msP[:, 0])[:2]
ax2.plot(neurtrans[:, 0], neurslp*neurtrans[:, 0]+neurint, color=cm(0.5))
if tsne:
    ax2.errorbar(tsnetrans[:, 0], msP[:, 0], xerr=tsnetrans[:, 1], yerr=msP[:, 1], color=cm(0.625), fmt='o', label=r'$\mathrm{t-SNE\enspace Spectral}$')
    tsneslp, tsneint = linregress(tsnetrans[:, 0], msP[:, 0])[:2]
    ax2.plot(tsnetrans[:, 0], tsneslp*tsnetrans[:, 0]+tsneint, color=cm(0.625))
ax2.set_xlabel(r'$T$')
ax2.set_ylabel(r'$P$')
ax2.legend(loc='upper left')

fig0.savefig('.'.join(base_pref+['pressure', 'png']))
fig1.savefig('.'.join(base_pref+['potential', 'png']))
fig2.savefig('.'.join(base_pref+['melting_curve', 'png']))