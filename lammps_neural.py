# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:31:27 2018

@author: Nicholas
"""

from __future__ import division, print_function
import sys, logging, pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from TanhScaler import TanhScaler
from sklearn.decomposition import PCA

mode = 'cpu'
nproc = 4

if mode == 'cpu':
    from sknn.platform import cpu64, threading
    if nproc == 2:
        from sknn.platform import threads2
    if nproc == 4:
        from sknn.platform import threads4
    if nproc == 6:
        from sknn.platform import threads6
    if nproc == 8:
        from sknn.platform import threads8
    if nproc == 12:
        from sknn.platform import threads12
    if nproc == 16:
        from sknn.platform import threads16
if mode == 'gpu':
    from sknn.platform import gpu64
    
from sknn.mlp import Classifier, Layer, Convolution, Native
from lasagne import layers as lasagne, nonlinearities as nl
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PIL import Image

# plotting parameters
plt.rc('text', usetex=True)
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
          'font.size': ftsz,
          'text.latex.preamble': r'\usepackage{amsmath}'r'\boldmath'}
plt.rcParams.update(params)
# element choice
try:
    el = sys.argv[1]
except:
    el = 'LJ'
# pressure
P = {'Ti': 2.0,
     'Al': 2.0,
     'Ni': 2.0,
     'Cu': 2.0,
     'LJ': 2.0}
# lattice type
lat = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}
# simulation name
name = 'remcmc'
# file prefix
prefix = '%s.%s.%s.%d.lammps' % (name, el.lower(), lat[el], int(P[el]))
# run details
property = 'radial_distribution'  # property for classification
n_dat = 64                        # number of datasets
ntrainsets = 12                   # number of training sets
scaler = 'tanh'                   # data scaling method
network = 'sknn_convolution_2d'   # neural network type
bpca = False                      # boolean for pca reduction
fit_func = 'logistic'             # fitting function
# summary of input
print('------------------------------------------------------------')
print('input summary')
print('------------------------------------------------------------')
print('potential: %s' % el.lower())
print('number of sets: ', n_dat)
print('property: ', property)
print('training sets (per phase): ', ntrainsets)
print('scaler: ', scaler)
print('network: ', network)
print('fitting function: ', fit_func)
print('pca reduction: ', str(bpca).lower())
print('------------------------------------------------------------')
# fitting functions
# the confidence interval for transition temps is best with logistic
def logistic(t, b, m):
    a = 0.0
    k = 1.0
    return a+np.divide(k, 1+np.exp(-b*(t-m)))
def gompertz(t, b, c):
    a = 1.0
    return a*np.exp(-b*np.exp(-c*t))
def richard(t, a, k, b, nu, q, m, c):
    return a+np.divide(k-a, np.power(c+q*np.exp(-b*(t-m)), 1./nu))
# initial fitting parameters
log_guess = [1.0, 0.5]
gomp_guess = [1.0, 1.0]
rich_guess = [0.0, 1.0, 3.0, 0.5, 0.5, 0.0, 1.0]
# fitting dictionaries
fit_funcs = {'logistic':logistic, 'richard':richard, 'gompertz':gompertz}
fit_guess = {'logistic':log_guess, 'richard':rich_guess, 'gompertz':gomp_guess}
print('fitting function defined')
print('------------------------------------------------------------')
# bounds for training data
lb = 0+ntrainsets
ub = n_dat-(ntrainsets+1)
# neural network construction
# simple dense network with relu activation
sknn_class = Classifier(layers=[Layer('Rectifier', units=64), Layer('Softmax')], learning_rate=2**-6, n_iter=256, random_state=0, verbose=True)
# 1d cnn - lasagne layers do not work currently
sknn_convol_1d = Classifier(layers=[Native(lasagne.Conv1DLayer, num_filters=4, filter_size=4, stride=1, pad=0, nonlinearity=nl.rectify),
                                    Native(lasagne.MaxPool1DLayer, pool_size=4),
                                    Native(lasagne.Conv1DLayer, num_filters=4, filter_size=4, stride=1, pad=0, nonlinearity=nl.rectify),
                                    Native(lasagne.MaxPool1DLayer, pool_size=4),
                                    Layer('Softmax')], learning_rate=2**-5, n_iter=64, random_state=0, verbose=True)
# 2d cnn using 1d filters of gradually decreasing size; input must be square (or transformable into square)
sknn_convol_2d = Classifier(layers=[Convolution('Rectifier', channels=4, kernel_shape=(8,1), kernel_stride=(1,1)),
                                    Convolution('Rectifier', channels=4, kernel_shape=(4,1), kernel_stride=(1,1)),
                                    Convolution('Rectifier', channels=4, kernel_shape=(2,1), kernel_stride=(1,1)),
                                    Convolution('Rectifier', channels=4, kernel_shape=(1,1), kernel_stride=(1,1)),
                                    Layer('Softmax')], learning_rate=2**-6, n_iter=256, random_state=0, verbose=True)
networks = {'sknn_classifier':sknn_class, 'sknn_convolution_1d':sknn_convol_1d, 'sknn_convolution_2d':sknn_convol_2d}
print('network initialized')
print('------------------------------------------------------------')
# load domains for rdf and sf
R = pickle.load(open(prefix+'.r.pickle'))[:]
Q = pickle.load(open(prefix+'.q.pickle'))[:]
# load simulation data
N = pickle.load(open(prefix+'.natoms.pickle'))
O = np.concatenate(tuple([i*np.ones(int(len(N)/n_dat), dtype=int) for i in xrange(n_dat)]), 0)
P = pickle.load(open(prefix+'.virial.pickle'))
trT = pickle.load(open(prefix+'.temp.pickle'))
T = np.concatenate(tuple([np.mean(trT[O == i])*np.ones(int(len(N)/n_dat), dtype=int) for i in xrange(n_dat)]), 0)
stT = np.concatenate(tuple([np.std(trT[O == i])*np.ones(int(len(N)/n_dat), dtype=int) for i in xrange(n_dat)]), 0)
G = pickle.load(open(prefix+'.rdf.pickle'))
S = pickle.load(open(prefix+'.sf.pickle'))
# sample space reduction for improving performance
smplspc = np.arange(0, N.size, 1)
N = N[smplspc]
O = O[smplspc]
P = P[smplspc]
trT = trT[smplspc]
T = T[smplspc]
stT = stT[smplspc]
G = G[smplspc]
S = S[smplspc]
print('data loaded')
print('------------------------------------------------------------')
# property dictionary
propdom = {'radial_distribution':R, 'structure_factor':Q}
properties = {'radial_distribution':G, 'structure_factor':S}
# scaler dict
scalers = {'standard':StandardScaler(), 'minmax':MinMaxScaler(feature_range=(0,1)), 'robust':RobustScaler(), 'tanh':TanhScaler()}
# pca initialization
npcacomp = np.shape(properties[property])[1]
pca = PCA(n_components=npcacomp)
print('scaler and pca reduction initialized')
print('------------------------------------------------------------')
# change pca properties for 2d convolution to prevent error
# 2d convolution expects square input
if network == 'sknn_convolution_2d':
    nrsqr = np.square(int(np.sqrt(np.shape(properties[property])[1]))) # nearest square to total observations
    pca.set_params(n_components=nrsqr) # set pca component parameter
# indices for partitioning data
sind = (O < lb)               # solid training indices
lind = (O > ub)               # liquid training indices
tind = (O < lb) | (O > ub)    # training indices
cind = (O >= lb) & (O <= ub)  # classification indices
# initialize data
data = properties[property]              # extract data from dictionary
scalers[scaler].fit(data[tind])          # fit scaler to training data
sdata = scalers[scaler].transform(data)  # transform data with scaler
# apply pca reduction
if bpca:
    pca.fit(sdata[tind])                  # pca fit to training data
    tdata = pca.transform(sdata[tind])    # pca transform training data
    cdata = pca.transform(sdata[cind])    # pca transform classification data
    evar = pca.explained_variance_ratio_  # extract explained variance ratios
# extract training/classification data/temperatures
if not bpca:
    if network == 'sknn_convolution_2d':
        tdata = sdata[tind, :nrsqr] # extract nearest square training data
        cdata = sdata[cind, :nrsqr] # extract nearest square classification data
    else:
        tdata = sdata[tind]  # extract training data
        cdata = sdata[cind]  # extract classification data
tT = T[tind]  # training temperatures
cT = T[cind]  # classification temperatures
ustdata = data[tind]  # unscaled training data
uscdata = data[cind]  # unscaled classification data
tshape = np.shape(tdata)
cshape = np.shape(cdata)
print('data scaled')
print('------------------------------------------------------------')
# display pca information
if bpca:
    print('data reduced')
    print('------------------------------------------------------------')
    print('pca fit information')
    print('------------------------------------------------------------')
    print('principal components: ', len(evar))
    print('explained variances: ', ', '.join(evar[:3].astype('|S32')), '...')
    print('total explained variance: ', np.sum(evar))
    print('------------------------------------------------------------')
# classification indices
tclass = np.array(np.count_nonzero(sind)*[0]+np.count_nonzero(lind)*[1], dtype=int)
# fit neural network to training data
networks[network].fit(tdata, tclass)
print('network fit to training data')
print('------------------------------------------------------------')
# classification of data
pred = networks[network].predict(cdata).reshape(-1)  # prediction classifications
prob = networks[network].predict_proba(cdata)        # prediction probabilities
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
# loop through temperature domain
for i in xrange(len(temps)):
    mprob[i] = np.mean(prob[cT == temps[i], 1])  # mean probability of samples at temp i being liquid
# curve fitting
adjtemps = np.linspace(0, 1, len(temps))                                                               # domain for curve fitting
n_dom = 4096                                                                                           # expanded number of curve samples
adjdom = np.linspace(0, 1, n_dom)                                                                      # expanded domain for curve fitting
fitdom = np.linspace(np.min(temps), np.max(temps), n_dom)                                                      # expanded domain for curve plotting
popt, pcov = curve_fit(fit_funcs[fit_func], adjtemps, mprob, p0=fit_guess[fit_func], method='dogbox')  # fitting parameters
perr = np.sqrt(np.diag(pcov))                                                                          # fit standard error
fitrng = fit_funcs[fit_func](adjdom, *popt)                                                            # fit values
# extract transition
if fit_func == 'gompertz':
    trans = -np.log(np.log(2)/popt[0])/popt[1]                                        # midpoint formula
    trans = trans*(np.max(temps)-np.min(temps))+np.min(temps)                         # transformation to temperature
    cerr = np.array([[-perr[0], perr[1]], [perr[0], -perr[1]]], dtype=float)          # critical error
    tintrvl = np.divide(-np.log(np.log(2)/(popt[0]+cerr[:, 0])), popt[1]+cerr[:, 1])  # error range
    tintrvl = tintrvl*(np.max(temps)-np.min(temps))+np.min(temps)                     # transformation to temperature interval
    cfitrng = [fit_funcs[fit_func](adjdom, *(popt+cerr[i, :])) for i in xrange(2)]    # critical fit values
if fit_func == 'logistic':
    trans = popt[1]*(np.max(temps)-np.min(temps))+np.min(temps)                        # midpoint temperature
    ferr = perr[1]*(np.max(temps)-np.min(temps))                                       # temperature fit error
    werr = 1/np.abs(temps-trans)                                                       # temperature error weight
    terr = np.sum(np.multiply(werr, stemps))/np.sum(werr)                              # temperature simulation error
    cerr = ferr+terr                                                                   # total critical error
    tintrvl = trans+cerr*np.array([-1, 1])                                             # temperature interval
    adjintrvl = (tintrvl-np.min(temps))/(np.max(temps)-np.min(temps))                  # adjusted interval
    cfitrng = [fit_funcs[fit_func](adjdom, popt[0], adjintrvl[i]) for i in xrange(2)]  # critical fit values
else:
    trans = adjdom[np.argmin(np.abs(fitrng-0.5))]
    trans = trans*(np.max(temps)-np.min(temps))+np.min(temps)
print('transition temperature estimated')
print('------------------------------------------------------------')
print('transition: ', trans)
print('transition range: ', ', '.join(tintrvl.astype('|S32')))
print('fit parameters: ', ', '.join(popt.astype('|S32')))
print('parameter error: ', ', '.join(perr.astype('|S32')))
print('------------------------------------------------------------')
# plot of phase probability
fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.xaxis.set_ticks_position('bottom')
ax0.yaxis.set_ticks_position('left')
ax0.plot(fitdom, fitrng, color=cm(scale(trans)), label='$\mathrm{Phase\enspace Probability\enspace Curve}$')
ax0.axvline(trans, color=cm(scale(trans)), alpha=0.50)
if fit_func == 'gompertz' or fit_func == 'logistic':
    for i in xrange(2):
        # ax0.plot(fitdom, cfitrng[i], color=cm(scale(tintrvl[i])), alpha=0.50, linestyle='--')
        ax0.axvline(tintrvl[i], color=cm(scale(tintrvl[i])), alpha=0.50, linestyle='--')
for i in xrange(2):
    ax0.scatter(cT[pred == i], prob[pred == i, 1], c=cm(scale(mtemp[1, i])), s=120, alpha=0.05, edgecolors='none')
ax0.scatter(temps, mprob, color=cm(scale(temps)), s=240, edgecolors='none', marker='*')
if el == 'LJ':
    ax0.text(trans+2*np.diff(temps)[0], .5, '$T_{\mathrm{trans}} = %.4f \pm %.4f$' % (trans, cerr))
else:
    ax0.text(trans+2*np.diff(temps)[0], .5, '$T_{\mathrm{trans}} = %4.0f \pm %4.0fK$' % (trans, cerr))
ax0.set_ylim(0.0, 1.0)
for tick in ax0.get_xticklabels():
    tick.set_rotation(16)
scitxt = ax0.yaxis.get_offset_text()
scitxt.set_x(.025)
ax0.legend(loc='lower right')
ax0.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
ax0.set_xlabel('$\mathrm{Temperature}$')
ax0.set_ylabel('$\mathrm{Probability}$')
ax0.set_title('$\mathrm{%s\enspace Phase\enspace Probabilities}$' % el, y=1.015)
# plot of trained and classified rdfs
labels = ['Solid', 'Liquid']
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
for i in xrange(2):
    plabels = ['$\mathrm{Trained\enspace %s\enspace Phase}$' % labels[i], '$\mathrm{Classified\enspace %s\enspace Phase}$' % labels[i]]
    ax1.plot(propdom[property], np.mean(ustdata[tclass == i], axis=0), color=cm(scale(mtemp[0, i])), alpha=1.00, label=plabels[0])
    ax1.plot(propdom[property], np.mean(uscdata[pred == i], axis=0), color=cm(scale(mtemp[1, i])), alpha=1.00, linestyle='--', label=plabels[1])
ax1.legend()
if property == 'radial_distribution':
    ax1.set_xlabel('$\mathrm{Distance}$')
    ax1.set_ylabel('$\mathrm{Radial Distribution}$')
    ax1.set_title('$\mathrm{%s\enspace Phase\enspace RDFs}$' % el, y=1.015)
if property == 'structure_factor':
    ax1.set_xlabel('$\mathrm{Wavenumber}$')
    ax1.set_ylabel('$\mathrm{Structure Factor}$')
    ax1.set_title('$\mathrm{%s\enspace Phase\enspace SFs}$' % el, y=1.015)
# prefix for plot files
if bpca:
    plt_pref = [prefix, network, property, scaler, 'reduced', fit_func]
else:
    plt_pref = [prefix, network, property, scaler, 'not-reduced', fit_func]
# generate convolution images
if 'sknn_convolution' in network:
    imgsd = int(np.sqrt(npcacomp))
    train0 = np.reshape(np.mean(tdata[tclass == 0], axis=0), 2*(imgsd,))
    train1 = np.reshape(np.mean(tdata[tclass == 1], axis=0), 2*(imgsd,))
    train0 = Image.fromarray(np.uint8(cm(train0)*255))
    train0 = train0.resize((400,400), Image.ANTIALIAS)
    train1 = Image.fromarray(np.uint8(cm(train1)*255))
    train1 = train1.resize((400,400), Image.ANTIALIAS)
    class0 = np.reshape(np.mean(cdata[pred == 0], axis=0), 2*(imgsd,))
    class1 = np.reshape(np.mean(cdata[pred == 1], axis=0), 2*(imgsd,))
    class0 = Image.fromarray(np.uint8(cm(class0)*255))
    class0 = class0.resize((400,400), Image.ANTIALIAS)
    class1 = Image.fromarray(np.uint8(cm(class1)*255))
    class1 = class1.resize((400,400), Image.ANTIALIAS)
    images = Image.new('RGB', (800,800))
    images.paste(train0, (0, 0))
    images.paste(train1, (400, 0))
    images.paste(class0, (0, 400))
    images.paste(class1, (400, 400))
    images.save('.'.join(plt_pref+['img.png']))
    
    print('images saved')
    print('------------------------------------------------------------')
# plt.show()
# save figures
fig0.savefig('.'.join(plt_pref+['prob.png']))
if property == 'radial_distribution':
    fig1.savefig('.'.join(plt_pref+['rdf', 'png']))
if property == 'structure_factor':
    fig1.savefig('.'.join(plt_pref+['sf', 'png']))
# close plots
plt.close('all')
print('plots saved')
print('------------------------------------------------------------')
print('finished')
print('------------------------------------------------------------')