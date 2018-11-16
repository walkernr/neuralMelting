# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:31:27 2018

@author: Nicholas
"""

import argparse
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from TanhScaler import TanhScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

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
PARSER.add_argument('-pn', '--pressure_number', help='number of pressures',
                    type=int, default=8)
PARSER.add_argument('-tn', '--temperature_number', help='number of temperatures',
                    type=int, default=48)
PARSER.add_argument('-sn', '--sample_number', help='sample number per temperature',
                    type=int, default=1024)
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
from keras.optimizers import SGD, Nadam
from keras.wrappers.scikit_learn import KerasRegressor

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
NP = ARGS.pressure_number     # number of pressure sets
NT = ARGS.temperature_number  # number of temperature sets
NS = ARGS.sample_number       # number of samples per temperature set
# data preparation parameters
EP = ARGS.epochs           # number of epochs
FTR = ARGS.feature         # feature to be learned
SCLR = ARGS.scaler         # feature scaler
RDCN = ARGS.reduction      # feature dimension reduction
# data analysis parameters
NN = ARGS.neural_network  # neural network

# lattice type
LAT = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}

# file prefix
PREFIXES = ['%s.%s.%s.%02d.lammps' % (NAME, EL.lower(), LAT[EL], i) for i in range(NP)]
# summary of input
if VERBOSE:
    print('------------------------------------------------------------')
    print('input summary')
    print('------------------------------------------------------------')
    print('potential:                   %s' % EL.lower())
    print('number of pressures:         %d' % NP)
    print('number of temperatures:      %d' % NT)
    print('number of samples:           %d' % NS)
    print('feature:                     %s' % FTR)
    print('scaler:                      %s' % SCLR)
    print('reduction:                   %s' % RDCN)
    print('network:                     %s' % NN)
    print('epochs:                      %d' % EP)
    print('------------------------------------------------------------')

# load simulation data
NATOMS = np.array([pickle.load(open(PREFIXES[i]+'.natoms.pickle', 'rb')).reshape(NT, NS) for i in range(NP)])
# load potential data
PE = np.array([pickle.load(open(PREFIXES[i]+'.pe.pickle', 'rb')).reshape(NT, NS) for i in range(NP)])
# load pressure data
VIRIAL = np.array([pickle.load(open(PREFIXES[i]+'.virial.pickle', 'rb')).reshape(NT, NS) for i in range(NP)])
# load temperature data
TEMP = np.array([pickle.load(open(PREFIXES[i]+'.temp.pickle', 'rb')).reshape(NT, NS) for i in range(NP)])
# load structure domains
R = np.array([pickle.load(open(PREFIXES[i]+'.r.pickle', 'rb')) for i in range(NP)])
Q = np.array([pickle.load(open(PREFIXES[i]+'.q.pickle', 'rb')) for i in range(NP)])
# load structure data
G = np.array([pickle.load(open(PREFIXES[i]+'.rdf.pickle', 'rb')).reshape(NT, NS, -1) for i in range(NP)])
S = np.array([pickle.load(open(PREFIXES[i]+'.sf.pickle', 'rb')).reshape(NT, NS, -1) for i in range(NP)])
I = np.array([pickle.load(open(PREFIXES[i]+'.ef.pickle', 'rb')).reshape(NT, NS, -1) for i in range(NP)])
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

DATA = FTRS[FTR].reshape(NP*NT*NS, -1)
T = TEMP.reshape(NP*NT*NS)
TRIN, TSIN, TROT, TSOT = train_test_split(DATA, T, test_size=0.5, random_state=SEED)

# data scaling
STRIN = SCLRS[SCLR].fit_transform(TRIN)
STSIN = SCLRS[SCLR].transform(TSIN)

if VERBOSE:
    print('data scaled')
    print('------------------------------------------------------------')

# reduction
RTRIN = RDCNS[RDCN].fit_transform(STRIN)
RTSIN = RDCNS[RDCN].transform(TSIN)

RTRIN = RTRIN[:, :, np.newaxis]
RTSIN = RTSIN[:, :, np.newaxis]

if VERBOSE:
    print('data reduced')
    print('------------------------------------------------------------')

# neural network construction
def build_keras_cnn1d():
    ''' builds 1-d convolutional neural network '''
    model = Sequential([Conv1D(filters=64, kernel_size=4, activation='relu', kernel_initializer='he_normal',
                               padding='causal', strides=1, input_shape=RTRIN.shape[1:]),
                        Conv1D(filters=64, kernel_size=4, activation='relu', kernel_initializer='he_normal',
                               padding='causal', strides=1),
                        MaxPooling1D(strides=4),
                        Conv1D(filters=128, kernel_size=4, activation='relu', kernel_initializer='he_normal',
                               padding='causal', strides=1),
                        Conv1D(filters=128, kernel_size=4, activation='relu', kernel_initializer='he_normal',
                               padding='causal', strides=1),
                        GlobalAveragePooling1D(),
                        Dropout(rate=0.5),
                        Dense(units=1, activation='sigmoid')])
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='mean_squared_error', optimizer=nadam, metrics=['accuracy'])
    return model


def build_keras_dense():
    model = Sequential([Dense(units=64, activation='relu', kernel_initializer='glorot_normal', input_dim=reduced_train_input.shape[1]),
                        Dense(units=64, activation='relu', kernel_initializer='glorot_normal'),
                        Dense(units=1, activation='linear', kernel_initializer='glorot_normal')])
    sgd = SGD(lr=0.004, momentum=0.99, decay=0.25, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

# NN = KerasRegressor(build_keras_cnn1d, epochs=EP, batch_size=8, verbose=VERBOSE)
NN = KerasRegressor(build_keras_dense, epochs=EP, batch_size=8, verbose=VERBOSE)

# cross validation
KFLD = KFold(n_splits=16, random_state=SEED)
RES = cross_val_score(NN, RTRIN, TROT, cv=KFLD)

# network training
NN.fit(RTRIN, TROT)

# network predictions
PRED = NN.predict(RTSIN)

# scores
RERR = np.divide(np.abs(TSOT-PRED), TSOT)
print('cross validation: %.2f (%.2f) mse' % (RES.mean(), RES.std()))
print('mse score: %.2f' % mean_squared_error(TSOT, PRED))
print('r2 score: %.2f' % r2_score(TSOT, PRED))
