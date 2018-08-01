# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:31:27 2018

@author: Nicholas
"""

import argparse
import pickle
import numpy as np
from scipy.stats import linregress
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# parse command line
PARSER = argparse.ArgumentParser()
PARSER.add_argument('-v', '--verbose', help='verbose output', action='store_true')
PARSER.add_argument('-n', '--name', help='name of simulation',
                    type=str, default='test_run')
PARSER.add_argument('-e', '--element', help='element choice',
                    type=str, default='LJ')
PARSER.add_argument('-pn', '--pressure_number', help='number of pressures',
                    type=int, default=4)
PARSER.add_argument('-tn', '--temperature_number', help='number of temperatures',
                    type=int, default=48)
PARSER.add_argument('-f', '--feature', help='feature to learn',
                    type=str, default='entropic_fingerprint')
PARSER.add_argument('-s', '--scaler', help='feature scaler',
                    type=str, default='tanh')
PARSER.add_argument('-r', '--reduction', help='dimension reduction method',
                    type=str, default='pca')
PARSER.add_argument('-nn', '--neural_network', help='neural network',
                    type=str, default='cnn1d')
PARSER.add_argument('-ff', '--fit_function', help='fitting function',
                    type=str, default='logistic')
# parse arguments
ARGS = PARSER.parse_args()
# verbosity
VERBOSE = ARGS.verbose
# simulation identifiers
NAME = ARGS.name              # simulation name
EL = ARGS.element             # element name
NP = ARGS.pressure_number     # pressure number
NT = ARGS.temperature_number  # number of temperature sets
# data preparation parameters
FTR = ARGS.feature         # feature to be learned
SCLR = ARGS.scaler         # feature scaler
RDCN = ARGS.reduction      # feature dimension reduction
# data analysis parameters
NN = ARGS.neural_network  # neural network
FF = ARGS.fit_function    # fitting function

# plotting parameters
plt.rc('font', family='sans-serif')
FTSZ = 48
PARAMS = {'figure.figsize': (26, 20),
          'lines.linewidth': 4.0,
          'legend.fontsize': FTSZ,
          'axes.labelsize': FTSZ,
          'axes.titlesize': FTSZ,
          'axes.linewidth': 2.0,
          'xtick.labelsize': FTSZ,
          'xtick.major.size': 20,
          'xtick.major.width': 2.0,
          'ytick.labelsize': FTSZ,
          'ytick.major.size': 20,
          'ytick.major.width': 2.0,
          'font.size': FTSZ}
plt.rcParams.update(PARAMS)

# lattice type
LAT = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}

# summary of input
if VERBOSE:
    print('------------------------------------------------------------')
    print('input summary')
    print('------------------------------------------------------------')
    print('potential:                   %s' % EL.lower())
    print('number of pressures:         %d' % NP)
    print('number of temperatures:      %d' % NT)
    print('feature:                     %s' % FTR)
    print('scaler:                      %s' % SCLR)
    print('reduction:                   %s' % RDCN)
    print('network:                     %s' % NN)
    print('fitting function:            %s' % FF)
    print('------------------------------------------------------------')


def extract_potential(i):
    u = pickle.load(open(PPREFS[i]+'.pe.pickle', 'rb')).reshape(NT, -1)
    um = np.mean(u, 1)
    us = np.std(u, 1)
    return um, us


def extract_pressure(i):
    p = pickle.load(open(PPREFS[i]+'.virial.pickle', 'rb')).reshape(NT, -1)
    pm = np.mean(p, 1)
    ps = np.std(p, 1)
    pms = np.array([np.mean(p), np.std(p)], np.float32)
    return pm, ps, pms


def extract_temperature(i):
    T = pickle.load(open(PPREFS[i]+'.temp.pickle', 'rb')).reshape(NT, -1)
    TM = np.mean(T, 1)
    TS = np.std(T, 1)
    return TM, TS


def extract_transition(i):
    trans = np.loadtxt(NPREFS[i]+'.out', dtype=np.float32)[0, :]
    return trans

# physical properties
UM = np.zeros((NP, NT), dtype=float)
US = np.zeros((NP, NT), dtype=float)
PM = np.zeros((NP, NT), dtype=float)
PS = np.zeros((NP, NT), dtype=float)
TM = np.zeros((NP, NT), dtype=float)
TS = np.zeros((NP, NT), dtype=float)
PMS = np.zeros((NP, 2), dtype=float)
TRANS = np.zeros((NP, 2), dtype=float)

# file prefixes
PREFIX = '%s.%s.%s.lammps' % (NAME, EL.lower(), LAT[EL])
PPREFS = ['%s.%s.%s.%02d.lammps' % (NAME, EL.lower(), LAT[EL], i) for i in range(NP)]
NPREFS = ['%s.%s.%s.%s.%s.%s' % (PPREFS[i], NN, FTR, SCLR, RDCN, FF) for i in range(NP)]

if VERBOSE:
    print('neural network transitions')
    print('pressure | temperature')
    print('------------------------------------------------------------')
for i in range(NP):
    UM[i, :], US[i, :] = extract_potential(i)
    PM[i, :], PS[i, :], PMS[i, :] = extract_pressure(i)
    TM[i, :], TS[i, :] = extract_temperature(i)
    TRANS[i, :] = extract_transition(i)
if VERBOSE:
    [print('%.2f %.2f' % (PMS[i, 0], TRANS[i, 0])) for i in range(NP)]
    print('------------------------------------------------------------')

CM = plt.get_cmap('plasma')
SCALE = lambda i: (PMS[i, 0]-np.min(PM))/np.max(PM)


def plot_pt():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(NP):
        ax.errorbar(TM[i], PM[i], xerr=TS[i], yerr=PS[i], color=CM(SCALE(i)), alpha=0.5,
                    label=r'$P = %.1f \pm %.1f$' % tuple(PMS[i]))
        ax.axvline(TRANS[i, 0], color=CM(SCALE(i)))
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$P$')
    ax.legend(loc='center right')
    fig.savefig(PREFIX+'.pt.png')


def plot_ut():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(NP):
        ax.errorbar(TM[i], UM[i], xerr=TS[i], yerr=US[i], color=CM(SCALE(i)), alpha=0.5,
                    label=r'$P = %.1f \pm %.1f$' % tuple(PMS[i]))
        ax.axvline(TRANS[i, 0], color=CM(SCALE(i)))
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$U$')
    ax.legend(loc='upper left')
    fig.savefig(PREFIX+'.ut.png')


def plot_mc():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if EL == 'LJ':
        # ref 1: http://paros.princeton.edu/cbe422/MP.pdf
        # ref 2: http://dirac.ruc.dk/~urp/interface.pdf
        litpress0 = np.array([1, 5, 10])
        litpress1 = np.array([0.928, 2.185, 3.514, 4.939, 7.921])
        littemp0 = np.array([0.77, 1.061, 1.379])
        littemp1 = np.array([0.7, 0.8, 0.9, 1.0, 1.2])
        litslp0, litint0 = linregress(littemp0, litpress0)[:2]
        litslp1, litint1 = linregress(littemp1, litpress1)[:2]
        ax.scatter(littemp0, litpress0, color=CM(0.25), s=240, edgecolors='none', marker='*',
                    label=r'$\mathrm{Literature\enspace (full\enspace potential)}$')
        ax.plot(littemp0, litslp0*littemp0+litint0, color=CM(0.25))
        ax.scatter(littemp1, litpress1, color=CM(0.375), s=240, edgecolors='none', marker='*',
                    label=r'$\mathrm{Literature\enspace} (r_c = 2.5)$')
        ax.plot(littemp1, litslp1*littemp1+litint1, color=CM(0.375))
    ax.errorbar(TRANS[:, 0], PMS[:, 0], xerr=TRANS[:, 1], yerr=PMS[:, 1], color=CM(0.5),
                fmt='o', label=r'$\mathrm{Keras\enspace CNN-1D}$')
    neurslp, neurint = linregress(TRANS[:, 0], PMS[:, 0])[:2]
    ax.plot(TRANS[:, 0], neurslp*neurtrans[:, 0]+neurint, color=CM(0.5))
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$P$')
    ax.legend(loc='upper left')
    fig.savefig(PREFIX+'.mc.png')

plot_pt()
plot_ut()
plot_mc()
