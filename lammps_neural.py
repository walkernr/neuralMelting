# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:31:27 2018

@author: Nicholas
"""

from __future__ import division, print_function
import os, sys, pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from TanhScaler import TanhScaler
from sklearn.decomposition import PCA, KernelPCA
from scipy.odr import ODR, Model, Data, RealData
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# number of threads
nthreads = 16

# keras backend
if '--theano' in sys.argv:
    os.environ['KERAS_BACKEND'] = 'theano'
else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# multithreading
os.environ['MKL_NUM_THREADS'] = str(nthreads)
os.environ['GOTO_NUM_THREADS'] = str(nthreads)
os.environ['OMP_NUM_THREADS'] = str(nthreads)
os.environ['openmp'] = 'True'

from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
from keras.optimizers import SGD, Nadam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import plot_model

seed = 256
set_random_seed(seed)

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
# property for classification
if '--property' in sys.argv:
    i = sys.argv.index('--property')
    property = sys.argv[i+1]
else:
    property = 'entropic_fingerprint'
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
# number of training sets
if '--ntrain' in sys.argv:
    i = sys.argv.index('--ntrain')
    ntrain = int(sys.argv[i+1])
else:
    ntrain = 8
# number of samples from each set
if '--nsmpl' in sys.argv:
    i = sys.argv.index('--nsmpl')
    nsmpl = int(sys.argv[i+1])
else:
    nsmpl = 1024
# data scaling method
if '--scaler' in sys.argv:
    i = sys.argv.index('--scaler')
    scaler = sys.argv[i+1]
else:
    scaler = 'tanh'
# neural network type
if '--network' in sys.argv:
    i = sys.argv.index('--network')
    network = sys.argv[i+1]
else:
    network = 'keras_cnn1d'
# reduction type
if '--reduction' in sys.argv:
    i = sys.argv.index('--reduction')
    reduc = sys.argv[i+1]
else:
    reduc = 'pca'
# fitting function
if '--fitfunc' in sys.argv:
    i = sys.argv.index('--fitfunc')
    fitfunc = sys.argv[i+1]
else:
    fitfunc = 'logistic'

# element and pressure index choice
if '--element' in sys.argv:
    i = sys.argv.index('--element')
    el = sys.argv[i+1]
else:
    el = 'LJ'
if '--pressindex' in sys.argv:
    i = sys.argv.index('--pressindex')
    pressind = int(sys.argv[i+1])
else:
    pressind = 0

# pressure
press = np.linspace(lpress, hpress, npress, dtype=np.float64)
# lattice type
lat = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}

# file prefix
prefix = '%s.%s.%s.%d.lammps' % (name, el.lower(), lat[el], int(press[pressind]))
# summary of input
print('------------------------------------------------------------')
print('input summary')
print('------------------------------------------------------------')
print('potential:                 %s' % el.lower())
print('pressure:                  %f' % press[pressind])  
print('number of sets:            %d' % ntemp)
print('number of samples:         %d' % nsmpl)
print('property:                  %s' % property)
print('training sets (per phase): %d' % ntrain)
print('scaler:                    %s' % scaler)
print('network:                   %s' % network)
print('fitting function:          %s' % fitfunc)
print('reduction:                 %s' % reduc)
print('------------------------------------------------------------')

# fitting functions
# the confidence interval for transition temps is best with logistic
def logistic(beta, t):
    a = 0.0
    k = 1.0
    b, m = beta
    return a+np.divide(k, 1+np.exp(-b*(t-m)))
def gompertz(beta, t):
    a = 1.0
    b, c = beta
    return a*np.exp(-b*np.exp(-c*t))
def richard(beta, t):
    a, k, b, nu, q, m, c = beta
    return a+np.divide(k-a, np.power(c+q*np.exp(-b*(t-m)), 1./nu))
# initial fitting parameters
log_guess = [1.0, 0.5]
gomp_guess = [1.0, 1.0]
rich_guess = [0.0, 1.0, 3.0, 0.5, 0.5, 0.0, 1.0]
# fitting dictionaries
fitfuncs = {'logistic':logistic, 'gompertz':gompertz, 'richard':richard}
fit_guess = {'logistic':log_guess, 'gompertz':gomp_guess, 'richard':rich_guess}
print('fitting function defined')
print('------------------------------------------------------------')

# load simulation data
N = pickle.load(open(prefix+'.natoms.pickle'))
O = np.concatenate(tuple([i*np.ones(int(len(N)/ntemp), dtype=int) for i in xrange(ntemp)]), 0)
# load potential data
trU = pickle.load(open(prefix+'.pe.pickle'))
U = np.concatenate(tuple([np.mean(trU[O == i])*np.ones(int(len(N)/ntemp), dtype=int) for i in xrange(ntemp)]), 0)
stU = np.concatenate(tuple([np.std(trU[O == i])*np.ones(int(len(N)/ntemp), dtype=int) for i in xrange(ntemp)]), 0)
# load pressure data
trP = pickle.load(open(prefix+'.virial.pickle'))
P = np.concatenate(tuple([np.mean(trP[O == i])*np.ones(int(len(N)/ntemp), dtype=int) for i in xrange(ntemp)]), 0)
stP = np.concatenate(tuple([np.std(trP[O == i])*np.ones(int(len(N)/ntemp), dtype=int) for i in xrange(ntemp)]), 0)
# load temperature data
trT = pickle.load(open(prefix+'.temp.pickle'))
T = np.concatenate(tuple([np.mean(trT[O == i])*np.ones(int(len(N)/ntemp), dtype=int) for i in xrange(ntemp)]), 0)
stT = np.concatenate(tuple([np.std(trT[O == i])*np.ones(int(len(N)/ntemp), dtype=int) for i in xrange(ntemp)]), 0)
# load structure domains
R = pickle.load(open(prefix+'.r.pickle'))[:]
Q = pickle.load(open(prefix+'.q.pickle'))[:]
# load structure data
G = pickle.load(open(prefix+'.rdf.pickle'))
S = pickle.load(open(prefix+'.sf.pickle'))
I = pickle.load(open(prefix+'.ef.pickle'))
# sample space reduction for improving performance
smplspc = np.concatenate(tuple([np.arange((i+1)*int(len(N)/ntemp)-nsmpl, (i+1)*int(len(N)/ntemp)) for i in xrange(ntemp)]))
N = N[smplspc]
O = O[smplspc]
trU = trU[smplspc]
U = U[smplspc]
stU = stU[smplspc]
trP = trP[smplspc]
P = P[smplspc]
stP = stP[smplspc]
trT = trT[smplspc]
T = T[smplspc]
stT = stT[smplspc]
G = G[smplspc]
S = S[smplspc]
I = I[smplspc]
print('data loaded')
print('------------------------------------------------------------')

# property dictionary
propdom = {'radial_distribution':R, 'entropic_fingerprint':R, 'structure_factor':Q}
properties = {'radial_distribution':G, 'entropic_fingerprint':I, 'structure_factor':S}

# scaler dict
scalers = {'standard':StandardScaler(), 'minmax':MinMaxScaler(feature_range=(0,1)), 'robust':RobustScaler(), 'tanh':TanhScaler()}
# pca initialization
npcacomp = properties[property].shape[1]
pca = PCA(n_components=npcacomp)
kpca = KernelPCA(n_components=npcacomp)
reducers = {'pca':pca, 'kpca':kpca}
print('scaler and reduction initialized')
print('------------------------------------------------------------')

# bounds for training data
lb = 0+ntrain
ub = ntemp-(ntrain+1)

# indices for partitioning data
sind = (O < lb)               # solid training indices
lind = (O > ub)               # liquid training indices
tind = (O < lb) | (O > ub)    # training indices
cind = (O >= lb) & (O <= ub)  # classification indices

# initialize data
data = properties[property]              # extract data from dictionary
scalers[scaler].fit(data[tind])          # fit scaler to training data
sdata = scalers[scaler].transform(data)  # transform data with scaler
print('data scaled')
print('------------------------------------------------------------')

# apply reduction and extract training/classification data
if reduc:
    reducers[reduc].fit(sdata[tind])                  # pca fit to training data
    tdata = reducers[reduc].transform(sdata[tind])    # pca transform training data
    cdata = reducers[reduc].transform(sdata[cind])    # pca transform classification data
    # display reduction information
    print('data reduced')
    print('------------------------------------------------------------')
    if reduc == 'pca':
        evar = pca.explained_variance_ratio_  # pca explained variance ratio
        print('pca fit information')
        print('------------------------------------------------------------')
        print('principal components:     %d' % len(evar))
        print('explained variances:      %s' % ', '.join(evar[:3].astype('|S32'))+', ...')
        print('total explained variance: %f' % np.sum(evar))
        print('------------------------------------------------------------')
else:
    tdata = sdata[tind]  # extract training data
    cdata = sdata[cind]  # extract classification data

tT = T[tind]  # training temperatures
cT = T[cind]  # classification temperatures
# reshape data for cnn1d
if 'cnn1d' in network:
    tdata = tdata[:, :, np.newaxis]
    cdata = cdata[:, :, np.newaxis]

ustdata = data[tind]  # unscaled training data
uscdata = data[cind]  # unscaled classification data
tshape = np.shape(tdata)
cshape = np.shape(cdata)
# classification indices
tclass = np.array(np.count_nonzero(sind)*[0]+np.count_nonzero(lind)*[1], dtype=int)

# neural network construction
# keras - dense
def build_keras_dense():
    model = Sequential([Dense(units=64, activation='relu', input_dim=tshape[1]),
                        Dropout(rate=0.5),
                        Dense(units=16, activation='relu'),
                        Dropout(rate=0.5),
                        Dense(units=8, activation='relu'),
                        Dropout(rate=0.5),
                        Dense(units=1, activation='softmax')])
    nadam = Nadam(lr=0.00024414062, beta_1=0.9375, beta_2=0.9990234375, epsilon=None, schedule_decay=0.00390625)
    model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
    return model
    
# keras - 1d cnn
def build_keras_cnn1d():    
    model = Sequential([Conv1D(filters=64, kernel_size=4, activation='relu', padding='causal', strides=1, input_shape=tshape[1:]),
                        Conv1D(filters=64, kernel_size=4, activation='relu', padding='causal', strides=1),
                        MaxPooling1D(strides=4),
                        Conv1D(filters=128, kernel_size=4, activation='relu', padding='causal', strides=1),
                        Conv1D(filters=128, kernel_size=4, activation='relu', padding='causal', strides=1),
                        GlobalAveragePooling1D(),
                        Dropout(rate=0.5),
                        Dense(units=1, activation='sigmoid')])
    nadam = Nadam(lr=0.0009765625, beta_1=0.9375, beta_2=0.9990234375, epsilon=None, schedule_decay=0.00390625)
    model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
    return model
    
keras_dense = KerasClassifier(build_keras_dense, epochs=2, verbose=True)
keras_cnn1d = KerasClassifier(build_keras_cnn1d, epochs=1, verbose=True)
networks = {'keras_dense':keras_dense, 'keras_cnn1d':keras_cnn1d}
print('network initialized')
print('------------------------------------------------------------')

# fit neural network to training data
networks[network].fit(tdata, tclass)
print('------------------------------------------------------------')
print('network fit to training data')
print('------------------------------------------------------------')

# classification of data
prob = networks[network].predict_proba(cdata)        # prediction probabilities
pred = prob[:, 1].round()                            # prediction classifications
print('------------------------------------------------------------')
print('network predicted classification data')
print('------------------------------------------------------------')

# extract indices of classes and mean temps
ind = []
mtemp = np.zeros((2, 2), dtype=float)
# loop through classes
for i in xrange(2):
    ind.append(np.where(pred == int(i)))    # indices of class i
    mtemp[0, i] = np.mean(tT[tclass == i])  # mean temp of training class i
    mtemp[1, i] = np.mean(cT[ind[i]])       # mean temp of class i
# colormap and color scaler
cm = plt.get_cmap('plasma')
scale = lambda temp: (temp-np.min(T))/np.max(T-np.min(T))
print('colormap and scale defined')
print('------------------------------------------------------------')

# curve fitting and transition temp extraction
temps = np.unique(cT)                      # temperature domain of classification data
stemps = np.unique(stT[cind])              # standard error of temperature domain
mprob = np.zeros(len(temps), dtype=float)  # mean probability array
sprob = np.zeros(len(temps), dtype=float)  # standard error porbability array
# loop through temperature domain
for i in xrange(len(temps)):
    mprob[i] = np.mean(prob[cT == temps[i], 1])  # mean probability of samples at temp i being liquid
    sprob[i] = np.std(prob[cT == temps[i], 1])   # standard error of samples at temp i being liquid
# curve fitting
odr_data = RealData(temps, mprob, stemps, sprob)
odr_model = Model(fitfuncs[fitfunc])
odr = ODR(odr_data, odr_model, fit_guess[fitfunc])
odr.set_job(fit_type=0)
fit_out = odr.run()
popt = fit_out.beta
perr = fit_out.sd_beta
if fitfunc == 'logistic':
	trans = popt[1]
	cerr = perr[1]
	tintrvl = trans+cerr*np.array([-1, 1])
if fitfunc == 'gompertz':
	trans = -np.log(np.log(2)/popt[0])/popt[1]
	cerr = np.array([[-perr[0], perr[1]], [perr[0], -perr[1]]], dtype=float)
	tintrvl = np.divide(-np.log(np.log(2)/(popt[0]+cerr[:, 0])), popt[1]+cerr[:, 1])
n_dom = 4096
fitdom = np.linspace(np.min(temps), np.max(temps), n_dom)
fitrng = fitfuncs[fitfunc](popt, fitdom)

print('transition temperature estimated')
print('------------------------------------------------------------')
print('transition:       %f, %f' % (trans, cerr))
print('transition range: %s' % ', '.join(tintrvl.astype('|S32')))
print('fit parameters:   %s' % ', '.join(popt.astype('|S32')))
print('parameter error:  %s' % ', '.join(perr.astype('|S32')))
print('------------------------------------------------------------')

# prefix for output files
if reduc:
    out_pref = [prefix, network, property, scaler, reduc, fitfunc]
else:
    out_pref = [prefix, network, property, scaler, 'none', fitfunc]

# save data to file
with open('.'.join(out_pref+[str(nsmpl), 'out']), 'w') as fo:
    fo.write('# -------------------------------------------------------------\n')
    fo.write('# parameters\n')
    fo.write('# ---------------------------------------------------------------\n')
    fo.write('# potential:                 %s\n' % el.lower())
    fo.write('# pressure:                  %f\n' % press[pressind])  
    fo.write('# number of sets:            %d\n' % ntemp)
    fo.write('# number of samples:         %d\n' % nsmpl)
    fo.write('# property:                  %s\n' % property)
    fo.write('# training sets (per phase): %d\n' % ntrain)
    fo.write('# scaler:                    %s\n' % scaler)
    fo.write('# network:                   %s\n' % network)
    fo.write('# fitting function:          %s\n' % fitfunc)
    fo.write('# reduction:                 %s\n' % reduc)
    fo.write('# ------------------------------------------------------------\n')
    fo.write('# transition | critical error\n')
    fo.write('%f %f\n' % (trans, cerr))
    fo.write('# transition interval\n')
    fo.write('%s\n' % ' '.join(tintrvl.astype('|S32')))
    fo.write('# optimal fit parameters\n')
    fo.write('%s\n' % ' '.join(popt.astype('|S32')))
    fo.write('# fit parameter error\n')
    fo.write('%s\n' % ' '.join(perr.astype('|S32')))

# plot of phase probability
fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.xaxis.set_ticks_position('bottom')
ax0.yaxis.set_ticks_position('left')
ax0.plot(fitdom, fitrng, color=cm(scale(trans)), label=r'$\mathrm{Phase\enspace Probability\enspace Curve}$')
ax0.axvline(trans, color=cm(scale(trans)), alpha=0.50)
if fitfunc == 'gompertz' or fitfunc == 'logistic':
    for i in xrange(2):
        # ax0.plot(fitdom, cfitrng[i], color=cm(scale(tintrvl[i])), alpha=0.50, linestyle='--')
        ax0.axvline(tintrvl[i], color=cm(scale(tintrvl[i])), alpha=0.50, linestyle='--')
for i in xrange(2):
    ax0.scatter(cT[pred == i], prob[pred == i, 1], c=cm(scale(mtemp[1, i])), s=120, alpha=0.05, edgecolors='none')
ax0.scatter(temps, mprob, color=cm(scale(temps)), s=240, edgecolors='none', marker='*')
if el == 'LJ':
    ax0.text(trans+2*np.diff(temps)[0], .5, r'$T_{\mathrm{trans}} = %.4f \pm %.4f$' % (trans, cerr))
else:
    ax0.text(trans+2*np.diff(temps)[0], .5, r'$T_{\mathrm{trans}} = %4.0f \pm %4.0fK$' % (trans, cerr))
ax0.set_ylim(0.0, 1.0)
for tick in ax0.get_xticklabels():
    tick.set_rotation(16)
scitxt = ax0.yaxis.get_offset_text()
scitxt.set_x(.025)
# ax0.legend(loc='upper left')
ax0.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
ax0.set_xlabel(r'$\mathrm{Temperature}$')
ax0.set_ylabel(r'$\mathrm{Probability}$')
# ax0.set_title(r'$\mathrm{%s\enspace Phase\enspace Probabilities}$' % el, y=1.015)

# plot of trained and classified rdfs
labels = ['Solid', 'Liquid']
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
for i in xrange(2):
    plabels = [r'$\mathrm{Trained\enspace %s\enspace Phase}$' % labels[i], r'$\mathrm{Classified\enspace %s\enspace Phase}$' % labels[i]]
    ax1.plot(propdom[property], np.mean(ustdata[tclass == i], axis=0), color=cm(scale(mtemp[0, i])), alpha=1.00, label=plabels[0])
    ax1.plot(propdom[property], np.mean(uscdata[pred == i], axis=0), color=cm(scale(mtemp[1, i])), alpha=1.00, linestyle='--', label=plabels[1])
ax1.legend()
if property == 'radial_distribution':
    ax1.set_xlabel(r'$\mathrm{Distance}$')
    ax1.set_ylabel(r'$\mathrm{Radial Distribution}$')
    # ax1.set_title(r'$\mathrm{%s\enspace Phase\enspace RDFs}$' % el, y=1.015)
if property == 'entropic_fingerprint':
    ax1.set_xlabel(r'$\mathrm{Distance}$')
    ax1.set_ylabel(r'$\mathrm{Entropic Fingerprint}$')
    # ax1.set_title(r'$\mathrm{%s\enspace Phase\enspace EFs}$' % el, y=1.015)
if property == 'structure_factor':
    ax1.set_xlabel(r'$\mathrm{Wavenumber}$')
    ax1.set_ylabel(r'$\mathrm{Structure Factor}$')
    # ax1.set_title(r'$\mathrm{%s\enspace Phase\enspace SFs}$' % el, y=1.015)

# network graph
if 'keras' in network:
    plot_model(networks[network].model, show_shapes=True, show_layer_names=True, to_file='.'.join(out_pref+['mdl', str(nsmpl), 'png']))

# save figures
fig0.savefig('.'.join(out_pref+['prob', str(nsmpl), 'png']))
fig1.savefig('.'.join(out_pref+['strf', str(nsmpl), 'png']))
# close plots
plt.close('all')
print('plots saved')
print('------------------------------------------------------------')
print('finished')
print('------------------------------------------------------------')