# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:31:27 2018

@author: Nicholas
"""

import argparse
import numpy as np
from scipy.stats import linregress
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# parse command line
PARSER = argparse.ArgumentParser()
PARSER.add_argument('-v', '--verbose', help='verbose output', action='store_true')
PARSER.add_argument('-n', '--name', help='name of simulation',
                    type=str, default='remcmc_init')
PARSER.add_argument('-e', '--element', help='element choice',
                    type=str, default='LJ')
PARSER.add_argument('-pn', '--pressure_number', help='number of pressures',
                    type=int, default=8)
PARSER.add_argument('-tn', '--temperature_number', help='number of temperatures',
                    type=int, default=48)
PARSER.add_argument('-sf', '--sfeature', help='supervised feature to learn',
                    type=str, default='entropic_fingerprint')
PARSER.add_argument('-uf', '--ufeature', help='unsupervised feature to learn',
                    type=str, default='entropic_fingerprint')
PARSER.add_argument('-ss', '--sscaler', help='supervised feature scaler',
                    type=str, default='tanh')
PARSER.add_argument('-us', '--uscaler', help='unsupervised feature scaler',
                    type=str, default='tanh')
PARSER.add_argument('-sr', '--sreduction', help='supervised dimension reduction method',
                    type=str, default='pca')
PARSER.add_argument('-ur', '--ureduction', help='unsupervised dimension reduction method',
                    type=str, default='tsne')
PARSER.add_argument('-nn', '--neural_network', help='neural network',
                    type=str, default='cnn1d')
PARSER.add_argument('-c', '--clustering', help='clustering method',
                    type=str, default='agglomerative')
PARSER.add_argument('-sff', '--sfit_function', help='supervised fitting function',
                    type=str, default='logistic')
PARSER.add_argument('-uff', '--ufit_function', help='unsupervised fitting function',
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
SFTR = ARGS.sfeature         # supervised feature to be learned
UFTR = ARGS.ufeature         # unsupervised feature to be learned
SSCLR = ARGS.sscaler         # supervised feature scaler
USCLR = ARGS.uscaler         # unsupervised feature scaler
SRDCN = ARGS.sreduction      # supervised feature dimension reduction
URDCN = ARGS.ureduction      # unsupervised feature dimension reduction
# data analysis parameters
NN = ARGS.neural_network  # neural network
CLST = ARGS.clustering    # clustering method
SFF = ARGS.sfit_function  # supervised fitting function
UFF = ARGS.ufit_function  # unsupervised fitting function

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
    print('sfeature:                    %s' % SFTR)
    print('ufeature:                    %s' % UFTR)
    print('sscaler:                     %s' % SSCLR)
    print('uscaler:                     %s' % USCLR)
    print('sreduction:                  %s' % SRDCN)
    print('ureduction:                  %s' % URDCN)
    print('network:                     %s' % NN)
    print('fitting sfunction:           %s' % SFF)
    print('fitting ufunction:           %s' % UFF)
    print('------------------------------------------------------------')


def extract_data(pref):
    ''' extracts data from neural network output '''
    data = np.loadtxt(pref+'.out', dtype=np.float32)
    trans = data[0]
    prob, pe, virial, temp = np.split(data[1:], 4, axis=0)
    mprob, sprob = np.split(prob, 2, axis=1)
    mpe, spe = np.split(pe, 2, axis=1)
    mvirial, svirial = np.split(virial, 2, axis=1)
    mtemp, stemp = np.split(temp, 2, axis=1)
    return trans, mprob.T, sprob.T, mpe.T, spe.T, mvirial.T, svirial.T, mtemp.T, stemp.T

# physical properties
TRANS = np.zeros((2, NP, 2), dtype=float)
MPROB = np.zeros((2, NP, NT), dtype=float)
SPROB = np.zeros((2, NP, NT), dtype=float)
MPE = np.zeros((2, NP, NT), dtype=float)
SPE = np.zeros((2, NP, NT), dtype=float)
MVIRIAL = np.zeros((2, NP, NT), dtype=float)
SVIRIAL = np.zeros((2, NP, NT), dtype=float)
MTEMP = np.zeros((2, NP, NT), dtype=float)
STEMP = np.zeros((2, NP, NT), dtype=float)

# file prefixes
PREFIX = '%s.%s.%s' % (NAME, EL.lower(), LAT[EL])
SPREFS = ['%s.%02d.lammps.%s.%s.%s.%s.%s' % (PREFIX, i,
                                             SFTR, SSCLR, SRDCN, NN, SFF) for i in range(NP)]
UPREFS = ['%s.%02d.lammps.%s.%s.%s.%s.%s' % (PREFIX, i,
                                             UFTR, USCLR, URDCN, NN, UFF) for i in range(NP)]

if VERBOSE:
    print('neural network transitions')
    print('pressure | temperature')
    print('------------------------------------------------------------')
for i in range(NP):
    (TRANS[0, i], MPROB[0, i], SPROB[0, i], MPE[0, i], SPE[0, i],
     MVIRIAL[0, i], SVIRIAL[0, i], MTEMP[0, i], STEMP[0, i]) = extract_data(SPREFS[i])
    (TRANS[1, i], MPROB[1, i], SPROB[1, i], MPE[1, i], SPE[1, i],
     MVIRIAL[1, i], SVIRIAL[1, i], MTEMP[1, i], STEMP[1, i]) = extract_data(UPREFS[i])
    if VERBOSE:
        print('%.2f %.2f %.2f %.2f' % (np.mean(MVIRIAL[0, i]), TRANS[0, i, 0],
                                       np.mean(MVIRIAL[1, i]), TRANS[1, i, 0]))
if VERBOSE:
    print('------------------------------------------------------------')

with open(PREFIX+'.pst', 'w') as fo:
    fo.write('# virial | virial standard error | transition | transition standard error\n')
    for i in range(NP):
        fo.write('%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f' % (np.mean(MVIRIAL[0, i]), TRANS[0, i, 0],
                                                              np.mean(SVIRIAL[0, i]), TRANS[0, i, 1],
                                                              np.mean(MVIRIAL[1, i]), TRANS[1, i, 0],
                                                              np.mean(SVIRIAL[1, i]), TRANS[1, i, 1]))
CM = plt.get_cmap('plasma')
SCALE = lambda i: (np.mean(MVIRIAL[0, i])-np.min(MVIRIAL[0]))/np.max(MVIRIAL[0])


def plot_pt():
    ''' plots the pressure as a function of temperature '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j in range(NP):
        ax.errorbar(MTEMP[j], MVIRIAL[j], xerr=STEMP[j], yerr=SVIRIAL[j], color=CM(SCALE(j)),
                    alpha=0.5,
                    label=r'$P = %.1f \pm %.1f$' % (np.mean(MVIRIAL[j]), np.mean(SVIRIAL[j])))
        ax.axvline(TRANS[j, 0], color=CM(SCALE(j)))
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$P$')
    ax.legend(loc='center right')
    fig.savefig(PREFIX+'.pt.png')


def plot_ut():
    ''' plots the potential energy as a function of temperature '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j in range(NP):
        ax.errorbar(MTEMP[j], MPE[j], xerr=STEMP[j], yerr=SPE[j], color=CM(SCALE(j)), alpha=0.5,
                    label=r'$P = %.1f \pm %.1f$' % (np.mean(MVIRIAL[j]), np.mean(SVIRIAL[j])))
        ax.axvline(TRANS[j, 0], color=CM(SCALE(j)))
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$U$')
    ax.legend(loc='upper left')
    fig.savefig(PREFIX+'.ut.png')


def plot_mc():
    ''' plots the melting curve '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if EL == 'LJ':
        # ref 1: http://paros.princeton.edu/cbe422/MP.pdf
        # ref 2: http://dirac.ruc.dk/~urp/interface.pdf
        rp0 = np.array([1, 5, 10])
        rp1 = np.array([0.928, 2.185, 3.514, 4.939, 7.921])
        rt0 = np.array([0.77, 1.061, 1.379])
        rt1 = np.array([0.7, 0.8, 0.9, 1.0, 1.2])
        rs0, ri0 = linregress(rt0, rp0)[:2]
        rs1, ri1 = linregress(rt1, rp1)[:2]
        ax.scatter(rt0, rp0, color=CM(0.25), s=240, edgecolors='none', marker='*',
                   label=r'$\mathrm{Literature\enspace (full\enspace potential)}$')
        ax.plot(rt0, rs0*rt0+ri0, color=CM(0.25))
        ax.scatter(rt1, rp1, color=CM(0.375), s=240, edgecolors='none', marker='*',
                   label=r'$\mathrm{Literature\enspace} (r_c = 2.5)$')
        ax.plot(rt1, rs1*rt1+ri1, color=CM(0.375))
    ax.errorbar(TRANS[:, 0], np.mean(MVIRIAL, axis=1), xerr=TRANS[:, 1],
                yerr=np.mean(SVIRIAL, axis=1), color=CM(0.5), fmt='o',
                label=r'$\mathrm{Keras\enspace CNN-1D}$')
    ns, ni = linregress(TRANS[:, 0], np.mean(MVIRIAL, axis=1))[:2]
    ax.plot(TRANS[:, 0], ns*TRANS[:, 0]+ni, color=CM(0.5))
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$P$')
    ax.legend(loc='upper left')
    fig.savefig(PREFIX+'.mc.png')

plot_pt()
plot_ut()
plot_mc()
