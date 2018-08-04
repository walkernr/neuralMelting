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
                    type=str, default='remcmc_run')
PARSER.add_argument('-e', '--element', help='element choice',
                    type=str, default='LJ')
PARSER.add_argument('-pn', '--pressure_number', help='number of pressures',
                    type=int, default=4)
PARSER.add_argument('-tn', '--temperature_number', help='number of temperatures',
                    type=int, default=96)
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


def extract_data(i):
    data = np.loadtxt(NPREFS[i]+'.out', dtype=np.float32)
    trans = data[0]
    prob, pe, virial, temp = np.split(data[1:], 4, axis=0)
    mprob, sprob = np.split(prob, 2, axis=1)
    mpe, spe = np.split(pe, 2, axis=1)
    mvirial, svirial = np.split(virial, 2, axis=1)
    mtemp, stemp = np.split(temp, 2, axis=1)
    return trans, mprob.T, sprob.T, mpe.T, spe.T, mvirial.T, svirial.T, mtemp.T, stemp.T

# physical properties
TRANS = np.zeros((NP, 2), dtype=float)
MPROB = np.zeros((NP, NT), dtype=float)
SPROB = np.zeros((NP, NT), dtype=float)
MPE = np.zeros((NP, NT), dtype=float)
SPE = np.zeros((NP, NT), dtype=float)
MVIRIAL = np.zeros((NP, NT), dtype=float)
SVIRIAL = np.zeros((NP, NT), dtype=float)
MTEMP = np.zeros((NP, NT), dtype=float)
STEMP = np.zeros((NP, NT), dtype=float)

# file prefixes
PREFIX = '%s.%s.%s.lammps' % (NAME, EL.lower(), LAT[EL])
NPREFS = ['%s.%s.%s.%02d.lammps.%s.%s.%s.%s.%s' % (NAME, EL.lower(), LAT[EL],
                                                   i, NN, FTR, SCLR, RDCN, FF) for i in range(NP)]

if VERBOSE:
    print('neural network transitions')
    print('pressure | temperature')
    print('------------------------------------------------------------')
for i in range(NP):
    (TRANS[i], MPROB[i], SPROB[i], MPE[i], SPE[i],
     MVIRIAL[i], SVIRIAL[i], MTEMP[i], STEMP[i]) = extract_data(i)
    if VERBOSE:
        print('%.2f %.2f' % (np.mean(MVIRIAL[i]), TRANS[i, 0]))
if VERBOSE:
    print('------------------------------------------------------------')

CM = plt.get_cmap('plasma')
SCALE = lambda i: (np.mean(MVIRIAL[i])-np.min(MVIRIAL))/np.max(MVIRIAL)


def plot_pt():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(NP):
        ax.errorbar(MTEMP[i], MVIRIAL[i], xerr=STEMP[i], yerr=SVIRIAL[i], color=CM(SCALE(i)), 
                    alpha=0.5,
                    label=r'$P = %.1f \pm %.1f$' % (np.mean(MVIRIAL[i]), np.mean(SVIRIAL[i])))
        ax.axvline(TRANS[i, 0], color=CM(SCALE(i)))
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$P$')
    ax.legend(loc='center right')
    fig.savefig(PREFIX+'.pt.png')


def plot_ut():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(NP):
        ax.errorbar(MTEMP[i], MPE[i], xerr=STEMP[i], yerr=SPE[i], color=CM(SCALE(i)), alpha=0.5,
                    label=r'$P = %.1f \pm %.1f$' % (np.mean(MVIRIAL[i]), np.mean(SVIRIAL[i])))
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
    ax.errorbar(TRANS[:, 0], np.mean(MVIRIAL, axis=1), xerr=TRANS[:, 1],
                yerr=np.mean(SVIRIAL, axis=1), color=CM(0.5), fmt='o',
                label=r'$\mathrm{Keras\enspace CNN-1D}$')
    neurslp, neurint = linregress(TRANS[:, 0], np.mean(MVIRIAL, axis=1))[:2]
    ax.plot(TRANS[:, 0], neurslp*TRANS[:, 0]+neurint, color=CM(0.5))
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$P$')
    ax.legend(loc='upper left')
    fig.savefig(PREFIX+'.mc.png')

plot_pt()
plot_ut()
plot_mc()
