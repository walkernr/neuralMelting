# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:31:27 2018

@author: Nicholas
"""

import argparse
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from scipy.odr import ODR, Model, RealData
from TanhScaler import TanhScaler

# parse command line
PARSER = argparse.ArgumentParser()
PARSER.add_argument('-v', '--verbose', help='verbose output', action='store_true')
PARSER.add_argument('-p', '--parallel', help='parallel run', action='store_true')
PARSER.add_argument('-pt', '--plot', help='plot results', action='store_true')
PARSER.add_argument('-nt', '--threads', help='number of threads',
                    type=int, default=16)
PARSER.add_argument('-b', '--backend', help='keras backend',
                    type=str, default='tensorflow')
PARSER.add_argument('-n', '--name', help='name of simulation',
                    type=str, default='remcmc_init')
PARSER.add_argument('-e', '--element', help='element choice',
                    type=str, default='LJ')
PARSER.add_argument('-i', '--pressure_index', help='pressure index',
                    type=int, default=0)
PARSER.add_argument('-tn', '--temperature_number', help='number of temperatures',
                    type=int, default=48)
PARSER.add_argument('-sn', '--sample_number', help='sample number per temperature',
                    type=int, default=1024)
PARSER.add_argument('-ln', '--learning_number', help='number of samples to learn per temperature',
                    type=int, default=1024)
PARSER.add_argument('-ts', '--training_sets', help='number of training sets per phase',
                    type=int, default=8)
PARSER.add_argument('-ep', '--epochs', help='number of epochs',
                    type=int, default=1)
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
# run specifications
VERBOSE = ARGS.verbose
PARALLEL = ARGS.parallel
PLOT = ARGS.plot
THREADS = ARGS.threads
BACKEND = ARGS.backend
# random seed
SEED = 256
np.random.seed(SEED)
# environment variables
os.environ['KERAS_BACKEND'] = BACKEND
if BACKEND == 'tensorflow':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow import set_random_seed
    set_random_seed(SEED)
if PARALLEL:
    os.environ['MKL_NUM_THREADS'] = str(THREADS)
    os.environ['GOTO_NUM_THREADS'] = str(THREADS)
    os.environ['OMP_NUM_THREADS'] = str(THREADS)
    os.environ['openmp'] = 'True'

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
from keras.optimizers import Nadam
from keras.wrappers.scikit_learn import KerasClassifier

if PLOT:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('font', family='sans-serif')
    FTSZ = 48
    PPARAMS = {'figure.figsize': (26, 20),
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
    plt.rcParams.update(PPARAMS)

# simulation identifiers
NAME = ARGS.name              # simulation name
EL = ARGS.element             # element name
PI = ARGS.pressure_index      # pressure index
NT = ARGS.temperature_number  # number of temperature sets
NS = ARGS.sample_number       # number of samples per temperature set
# data preparation parameters
LN = ARGS.learning_number  # number of learning samples per temperature set
TS = ARGS.training_sets    # number of training sets per phase
EP = ARGS.epochs           # number of epochs
FTR = ARGS.feature         # feature to be learned
SCLR = ARGS.scaler         # feature scaler
RDCN = ARGS.reduction      # feature dimension reduction
# data analysis parameters
NN = ARGS.neural_network  # neural network
FF = ARGS.fit_function    # fitting function
# training and classification sample counts
TST = 2*TS     # number of training sets
CS = NT-TST    # number of classification sets
SN = NT*LN     # total samples
TSN = TS*LN    # training samples for single phase
TSTN = TST*LN  # total training samples
CSN = CS*LN    # total classification samples
# training indices
TI = np.arange(-TS, TS)

# lattice type
LAT = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}

EPS = np.finfo(np.float32).eps

# file prefix
PREFIX = '%s.%s.%s.%02d.lammps' % (NAME, EL.lower(), LAT[EL], PI)
# summary of input
if VERBOSE:
    print('------------------------------------------------------------')
    print('input summary')
    print('------------------------------------------------------------')
    print('potential:                   %s' % EL.lower())
    print('pressure index:              %d' % PI)
    print('number of sets:              %d' % NT)
    print('number of samples (per set): %d' % NS)
    print('training sets (per phase):   %d' % TS)
    print('training samples (per set):  %d' % LN)
    print('feature:                     %s' % FTR)
    print('scaler:                      %s' % SCLR)
    print('reduction:                   %s' % RDCN)
    print('network:                     %s' % NN)
    print('epochs:                      %d' % EP)
    print('fitting function:            %s' % FF)
    print('------------------------------------------------------------')


# fitting functions
def logistic(beta, t):
    ''' returns logistic sigmoid '''
    a = 0.0
    k = 1.0
    b, m = beta
    return a+np.divide(k, 1+np.exp(-b*(t-m)))

# fit function dictionary
FFS = {'logistic': logistic}
# initial fit parameter dictionary
FGS = {'logistic': [1.0, 0.5]}
if VERBOSE:
    print('fitting function defined')
    print('------------------------------------------------------------')

# load simulation data
NATOMS = pickle.load(open(PREFIX+'.natoms.pickle', 'rb')).reshape(NT, NS)
# load potential data
PE = pickle.load(open(PREFIX+'.pe.pickle', 'rb')).reshape(NT, NS)
# load pressure data
VIRIAL = pickle.load(open(PREFIX+'.virial.pickle', 'rb')).reshape(NT, NS)
# load temperature data
TEMP = pickle.load(open(PREFIX+'.temp.pickle', 'rb')).reshape(NT, NS)
# load structure domains
R = pickle.load(open(PREFIX+'.r.pickle', 'rb'))
Q = pickle.load(open(PREFIX+'.q.pickle', 'rb'))
# load structure data
G = pickle.load(open(PREFIX+'.rdf.pickle', 'rb')).reshape(NT, NS, -1)
S = pickle.load(open(PREFIX+'.sf.pickle', 'rb')).reshape(NT, NS, -1)
I = pickle.load(open(PREFIX+'.ef.pickle', 'rb')).reshape(NT, NS, -1)
# sample space reduction for improving performance
NATOMS = NATOMS[:, -LN:]
PE = PE[:, -LN:]
VIRIAL = VIRIAL[:, -LN:]
TEMP = TEMP[:, -LN:]
G = G[:, -LN:, :]
S = S[:, -LN:, :]
I = I[:, -LN:, :]
if VERBOSE:
    print('data loaded')
    print('------------------------------------------------------------')

# property dictionary
FDOM = {'radial_distribution': R,
        'entropic_fingerprint': R,
        'structure_factor': Q}
FTRS = {'radial_distribution': G,
        'entropic_fingerprint': I,
        'structure_factor': S}

# scaler dictionary
SCLRS = {'standard':StandardScaler(),
         'minmax':MinMaxScaler(feature_range=(0, 1)),
         'robust':RobustScaler(),
         'tanh':TanhScaler()}
# reducers dictionary
NPCA = FTRS[FTR].shape[-1]
RDCNS = {'pca':PCA(n_components=NPCA),
         'kpca':KernelPCA(n_components=NPCA, n_jobs=THREADS),
         'isomap':Isomap(n_components=NPCA, n_jobs=THREADS),
         'lle':LocallyLinearEmbedding(n_components=NPCA, n_jobs=THREADS)}
if VERBOSE:
    print('scaler and reduction initialized')
    print('------------------------------------------------------------')

# initialize data
DATA = FTRS[FTR]                                     # extract data from dictionary
SCLRS[SCLR].fit(DATA[TI].reshape(TSTN, -1))          # fit scaler to training data
SDATA = SCLRS[SCLR].transform(DATA.reshape(SN, -1))  # transform data with scaler
if VERBOSE:
    print('data scaled')
    print('------------------------------------------------------------')

# apply reduction and extract training/classification data
if RDCN != 'none':
    RDCNS[RDCN].fit(SDATA.reshape(NT, NS, -1)[TI].reshape(TSTN, -1))  # pca fit to training data
    RDATA = RDCNS[RDCN].transform(SDATA)                              # pca transform training data
    # display reduction information
    if VERBOSE:
        print('data reduced')
        print('------------------------------------------------------------')
        if RDCN == 'pca':
            EVAR = RDCNS[RDCN].explained_variance_ratio_  # pca explained variance ratio
            print('pca fit information')
            print('------------------------------------------------------------')
            print('principal components:     %d' % len(EVAR))
            print('explained variances:      %f %f %f ...' % tuple(EVAR[:3]))
            print('total explained variance: %f' % np.sum(EVAR))
            print('------------------------------------------------------------')
else:
    RDATA = SDATA
# training and classification data
TDATA = RDATA.reshape(NT, NS, -1)[TI].reshape(TSTN, -1)
CDATA = RDATA.reshape(NT, NS, -1)[TS:-TS].reshape(CSN, -1)
TTEMP = TEMP[TI].reshape(TSTN)
CTEMP = TEMP[TS:-TS].reshape(CSN)

# reshape data for cnn1d
if 'cnn1d' in NN:
    TDATA = TDATA[:, :, np.newaxis]
    CDATA = CDATA[:, :, np.newaxis]

UTDATA = DATA[TI].reshape(TSTN, -1)     # unscaled training data
UCDATA = DATA[TS:-TS].reshape(CSN, -1)  # unscaled classification data
TSHP = TDATA.shape
CSHP = CDATA.shape
# classification indices
TC = np.concatenate((np.ones(TSN, dtype=np.uint16), np.zeros(TSN, dtype=np.uint16)), 0)


# neural network construction
def build_keras_cnn1d():
    ''' builds 1-d convolutional neural network '''
    model = Sequential([Conv1D(filters=64, kernel_size=4, activation='relu',
                               padding='causal', strides=1, input_shape=TSHP[1:]),
                        Conv1D(filters=64, kernel_size=4, activation='relu',
                               padding='causal', strides=1),
                        MaxPooling1D(strides=4),
                        Conv1D(filters=128, kernel_size=4, activation='relu',
                               padding='causal', strides=1),
                        Conv1D(filters=128, kernel_size=4, activation='relu',
                               padding='causal', strides=1),
                        GlobalAveragePooling1D(),
                        Dropout(rate=0.5),
                        Dense(units=1, activation='sigmoid')])
    nadam = Nadam(lr=0.0009765625, beta_1=0.9375, beta_2=0.9990234375,
                  epsilon=None, schedule_decay=0.00390625)
    model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
    return model

# network dictionary
NNS = {'cnn1d':KerasClassifier(build_keras_cnn1d, epochs=EP, verbose=VERBOSE)}
if VERBOSE:
    print('network initialized')
    print('------------------------------------------------------------')

# fit neural network to training data
NNS[NN].fit(TDATA, TC)
if VERBOSE:
    print('------------------------------------------------------------')
    print('network fit to training data')
    print('------------------------------------------------------------')

# classification of data
PROB = NNS[NN].predict_proba(CDATA)[:, 1]  # prediction probabilities
CC = PROB.round()                          # prediction classifications
if VERBOSE:
    print('------------------------------------------------------------')
    print('network predicted classification data')
    print('------------------------------------------------------------')

# reshape data
TTEMP = TTEMP.reshape(TST, LN)
CTEMP = CTEMP.reshape(CS, LN)
PROB = PROB.reshape(CS, LN)
TC = TC.reshape(TST, LN)
CC = CC.reshape(CS, LN)

# mean temps
MTEMP = np.array([[np.mean(TTEMP[TC == i]) for i in range(2)],
                  [np.mean(CTEMP[CC == i]) for i in range(2)]], dtype=np.float32)
# curve fitting and transition temp extraction
TDOM = np.mean(TEMP, 1)                                                        # temp domain
TERR = np.std(TEMP, 1)                                                         # temp standard error
MPROB = np.concatenate((np.zeros(TS), np.mean(PROB, 1), np.ones(TS)), 0)       # mean prob
SPROB = np.concatenate((EPS*np.ones(TS), np.std(PROB, 1), EPS*np.ones(TS)), 0) # prob standard error
# curve fitting
ODR_DATA = RealData(TDOM, MPROB, TERR, SPROB)
ODR_MODEL = Model(FFS[FF])
ODR_ = ODR(ODR_DATA, ODR_MODEL, FGS[FF])
ODR_.set_job(fit_type=0)
FIT = ODR_.run()
POPT = FIT.beta
PERR = FIT.sd_beta
TRANS = POPT[1]
CERR = PERR[1]
TINT = TRANS+CERR*np.array([-1, 1])
NDOM = 4096
FITDOM = np.linspace(np.min(TDOM), np.max(TDOM), NDOM)
FITVAL = FFS[FF](POPT, FITDOM)

if VERBOSE:
    print('transition temperature estimated')
    print('------------------------------------------------------------')
    print('transition:       %f %f' % (TRANS, CERR))
    print('transition range: %s %s' % tuple(TINT))
    print('fit parameters:   %s %s' % tuple(POPT))
    print('parameter error:  %s %s' % tuple(PERR))
    print('------------------------------------------------------------')

# prefix for output files
OUTPREF = '%s.%s.%s.%s.%s.%s' % (PREFIX, FTR, SCLR, RDCN, NN, FF)

# save data to file
with open(OUTPREF+'.out', 'w') as output:
    output.write('# -------------------------------------------------------------\n')
    output.write('# parameters\n')
    output.write('# ---------------------------------------------------------------\n')
    output.write('# potential:                   %s\n' % EL.lower())
    output.write('# pressure index:              %d\n' % PI)
    output.write('# number of sets:              %d\n' % NT)
    output.write('# number of samples (per set): %d\n' % NS)
    output.write('# training sets (per phase):   %d\n' % TS)
    output.write('# training samples (per set):  %d\n' % LN)
    output.write('# property:                    %s\n' % FTR)
    output.write('# scaler:                      %s\n' % SCLR)
    output.write('# reduction:                   %s\n' % RDCN)
    output.write('# network:                     %s\n' % NN)
    output.write('# epochs:                      %d\n' % EP)
    output.write('# fitting function:            %s\n' % FF)
    output.write('# ------------------------------------------------------------\n')
    output.write('# transition | standard error\n')
    output.write('%f %f\n' % (TRANS, CERR))
    output.write('# liquid probability | standard error\n')
    for i in range(NT):
        output.write('%f %f\n' % (MPROB[i], SPROB[i]))
    output.write('# potential | standard error\n')
    for i in range(NT):
        output.write('%f %f\n' % (np.mean(PE[i, :]), np.std(PE[i, :])))
    output.write('# virial | standard error\n')
    for i in range(NT):
        output.write('%f %f\n' % (np.mean(VIRIAL[i, :]), np.std(VIRIAL[i, :])))
    output.write('# temperature | standard error\n')
    for i in range(NT):
        output.write('%f %f\n' % (np.mean(TEMP[i, :]), np.std(TEMP[i, :])))


def plot_phase_probs():
    ''' plot of phase probability '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.plot(FITDOM, FITVAL, color=CM(SCALE(TRANS)),
            label=r'$\mathrm{Phase\enspace Probability\enspace Curve}$')
    ax.axvline(TRANS, color=CM(SCALE(TRANS)), alpha=0.50)
    for j in range(2):
        ax.axvline(TINT[j], color=CM(SCALE(TINT[j])), alpha=0.50, linestyle='--')
    for j in range(2):
        ax.scatter(CTEMP[CC == j], PROB[CC == j], c=CM(SCALE(MTEMP[1, j])),
                   s=120, alpha=0.05, edgecolors='none')
        ax.scatter(TTEMP[TC == j], j*np.ones(TC[TC == j].shape), c=CM(SCALE(MTEMP[0, j])),
                   s=120, alpha=0.05, edgecolors='none')
    ax.scatter(TDOM, MPROB, color=CM(SCALE(TDOM)), s=240, edgecolors='none', marker='*')
    if EL == 'LJ':
        ax.text(TRANS+2*np.diff(TDOM)[0], .5,
                r'$T_{\mathrm{trans}} = %.4f \pm %.4f$' % (TRANS, CERR))
    else:
        ax.text(TRANS+2*np.diff(TDOM)[0], .5,
                r'$T_{\mathrm{trans}} = %4.0f \pm %4.0fK$' % (TRANS, CERR))
    ax.set_ylim(0.0, 1.0)
    for tick in ax.get_xticklabels():
        tick.set_rotation(16)
    scitxt = ax.yaxis.get_offset_text()
    scitxt.set_x(.025)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.set_xlabel(r'$\mathrm{Temperature}$')
    ax.set_ylabel(r'$\mathrm{Probability}$')
    fig.savefig(OUTPREF+'.prob.png')


def plot_ftrs():
    ''' plot of trained and classified features '''
    labels = ['Solid', 'Liquid']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    for j in range(2):
        plabels = [r'$\mathrm{Trained\enspace %s\enspace Phase}$' % labels[j],
                   r'$\mathrm{Classified\enspace %s\enspace Phase}$' % labels[j]]
        ax.plot(FDOM[FTR], np.mean(UTDATA[TC.reshape(-1) == j], axis=0),
                color=CM(SCALE(MTEMP[0, j])), alpha=1.00, label=plabels[0])
        ax.plot(FDOM[FTR], np.mean(UCDATA[CC.reshape(-1) == j], axis=0),
                color=CM(SCALE(MTEMP[1, j])), alpha=1.00, linestyle='--', label=plabels[1])
    ax.legend()
    if property == 'radial_distribution':
        ax.set_xlabel(r'$\mathrm{Distance}$')
        ax.set_ylabel(r'$\mathrm{Radial Distribution}$')
    if property == 'entropic_fingerprint':
        ax.set_xlabel(r'$\mathrm{Distance}$')
        ax.set_ylabel(r'$\mathrm{Entropic Fingerprint}$')
    if property == 'structure_factor':
        ax.set_xlabel(r'$\mathrm{Wavenumber}$')
        ax.set_ylabel(r'$\mathrm{Structure Factor}$')
    fig.savefig(OUTPREF+'.ftr.png')

# save figures
if PLOT:
    # colormap and color scaler
    CM = plt.get_cmap('plasma')
    SCALE = lambda temp: (temp-np.min(TEMP))/np.max(TEMP-np.min(TEMP))
    if VERBOSE:
        print('colormap and scale defined')
        print('------------------------------------------------------------')
    # plots
    # plot_model(NNS[NN].model, show_shapes=True, show_layer_names=True, to_file=OUTPREF+'.mdl.png')
    plot_phase_probs()
    plot_ftrs()
    if VERBOSE:
        print('plots saved')
        print('------------------------------------------------------------')
if VERBOSE:
    print('finished')
    print('------------------------------------------------------------')
