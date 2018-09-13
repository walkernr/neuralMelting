# -*- coding: utf-8 -*-
"""
Created on Sat May 26 01:22:43 2018

@author: Nicholas
"""

import argparse
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from TanhScaler import TanhScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from scipy.odr import ODR, Model, RealData

# parse command line
PARSER = argparse.ArgumentParser()
PARSER.add_argument('-v', '--verbose', help='verbose output', action='store_true')
PARSER.add_argument('-pt', '--plot', help='plot results', action='store_true')
PARSER.add_argument('-nt', '--threads', help='number of threads',
                    type=int, default=16)
PARSER.add_argument('-n', '--name', help='name of simulation',
                    type=str, default='remcmc_init')
PARSER.add_argument('-e', '--element', help='element choice',
                    type=str, default='LJ')
PARSER.add_argument('-i', '--pressure_index', help='pressure index',
                    type=int, default=0)
PARSER.add_argument('-tn', '--temperature_number', help='number of temperatures',
                    type=int, default=96)
PARSER.add_argument('-sn', '--sample_number', help='sample number per temperature',
                    type=int, default=1024)
PARSER.add_argument('-ln', '--learning_number', help='number of samples to learn per temperature',
                    type=int, default=1024)
PARSER.add_argument('-f', '--feature', help='feature to learn',
                    type=str, default='entropic_fingerprint')
PARSER.add_argument('-s', '--scaler', help='feature scaler',
                    type=str, default='tanh')
PARSER.add_argument('-r', '--reduction', help='dimension reduction method',
                    type=str, default='tsne')
PARSER.add_argument('-c', '--clustering', help='clustering method',
                    type=str, default='spectral')
PARSER.add_argument('-ff', '--fit_function', help='fitting function',
                    type=str, default='logistic')
# parse arguments
ARGS = PARSER.parse_args()
# run specifications
VERBOSE = ARGS.verbose
PLOT = ARGS.plot
THREADS = ARGS.threads
# simulation identifiers
NAME = ARGS.name              # simulation name
EL = ARGS.element             # element name
PI = ARGS.pressure_index      # pressure index
NT = ARGS.temperature_number  # number of temperature sets
NS = ARGS.sample_number       # number of samples per temperature set
# data preparation parameters
LN = ARGS.learning_number  # number of learning samples per temperature set
FTR = ARGS.feature         # feature to be learned
SCLR = ARGS.scaler         # feature scaler
RDCN = ARGS.reduction      # feature dimension reduction
# data analysis parameters
CLST = ARGS.clustering  # clustering method
FF = ARGS.fit_function  # fitting function

if PLOT:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
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
    print('training samples (per set):  %d' % LN)
    print('feature:                     %s' % FTR)
    print('scaler:                      %s' % SCLR)
    print('reduction:                   %s' % RDCN)
    print('clustering:                  %s' % CLST)
    print('fitting function:            %s' % FF)
    print('------------------------------------------------------------')


# fitting functions
def logistic(beta, t):
    ''' returns logistic sigmoid '''
    a = 0.0
    k = 1.0
    b, m = beta
    return a+np.divide(k, 1+np.exp(-b*(t-m)))


def gompertz(beta, t):
    ''' returns gompertz sigmoid '''
    a = 1.0
    b, c = beta
    return a*np.exp(-b*np.exp(-c*t))

# fit function dictionary
FFS = {'logistic': logistic,
       'gompertz': gompertz}
# initial fit parameter dictionary
FGS = {'logistic': [16.0, 0.5],
       'gompertz': [1.0, 1.0]}
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
# dimensionality
NPCA = 64  # FTRS[FTR].shape[-1]
NNLN = 2
# temperature distribution
PLXTY= NS
RDCNS = {'pca':PCA(n_components=NPCA),
         'kpca':KernelPCA(n_components=NNLN, n_jobs=THREADS),
         'isomap':Isomap(n_components=NNLN, n_jobs=THREADS),
         'lle':LocallyLinearEmbedding(n_components=NNLN, n_jobs=THREADS),
         'tsne':TSNE(n_components=NNLN, perplexity=PLXTY, verbose=True, n_jobs=THREADS)}
CLSTS = {'agglom': AgglomerativeClustering(n_clusters=2),
         'kmeans': KMeans(n_jobs=THREADS, n_clusters=2, init='k-means++'),
         'spectral': SpectralClustering(n_jobs=THREADS, n_clusters=2)}
if VERBOSE:
    print('scaler, reduction, and clustering initialized')
    print('------------------------------------------------------------')

# scale data
SDATA = SCLRS[SCLR].fit_transform(FTRS[FTR].reshape(NT*LN, -1))
if VERBOSE:
    print('data scaled')
    print('------------------------------------------------------------')

# apply reduction and extract training/classification data
PDATA = RDCNS['pca'].fit_transform(SDATA)  # pca reduction
# display reduction information
if VERBOSE:
    print('data reduced')
    print('------------------------------------------------------------')
    EVAR = RDCNS['pca'].explained_variance_ratio_  # pca explained variance ratio
    print('pca fit information')
    print('------------------------------------------------------------')
    print('principal components:     %d' % len(EVAR))
    print('explained variances:      %f %f %f ...' % tuple(EVAR[:3]))
    print('total explained variance: %f' % np.sum(EVAR))
    print('------------------------------------------------------------')
if RDCN != 'none':
    RDATA = RDCNS[RDCN].fit_transform(PDATA)  # nonlinear reduction
else:
    RDATA = PDATA[:, :NNLN]

# clustering classification prediction
PRED = CLSTS[CLST].fit_predict(RDATA)
if CLST == 'kmeans':
    print('kmeans fit information')
    print('------------------------------------------------------------')
    print('intertia: ', CLSTS[CLST].inertia_)
    print('------------------------------------------------------------')    

# cluster mean temp
CMT = [np.mean(TEMP[PRED.reshape(NT, LN) == i]) for i in range(2)]
if CMT[0] > CMT[1]:
    PRED = 1-PRED
    CMT = np.flip(CMT)
# extract classification probabilities
MPRED = np.mean(PRED.reshape(NT, LN), 1)
SPRED = np.std(PRED.reshape(NT, LN), 1)+EPS

# curve fitting and transition temp extraction
TDOM = np.mean(TEMP, 1)  # temp domain
TERR = np.std(TEMP, 1)   # temp standard error
# curve fitting
ODR_DATA = RealData(TDOM, MPRED, TERR, SPRED)
ODR_MODEL = Model(FFS[FF])
ODR_ = ODR(ODR_DATA, ODR_MODEL, FGS[FF])
ODR_.set_job(fit_type=0)
FIT = ODR_.run()
POPT = FIT.beta
PERR = FIT.sd_beta
if FF == 'logistic':
    TRANS = POPT[1]
    CERR = PERR[1]
    TINT = TRANS+CERR*np.array([-1, 1])
if FF == 'gompertz':
    TRANS = -np.log(np.log(2)/POPT[0])/POPT[1]
    CERR = np.array([[-PERR[0], PERR[1]], [PERR[0], -PERR[1]]], dtype=np.float32)
    TINT = np.divide(-np.log(np.log(2)/(POPT[0]+CERR[:, 0])), POPT[1]+CERR[:, 1])
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

OUTPREF = '%s.%s.%s.%s.%s.%s' % (PREFIX, FTR, SCLR, RDCN, CLST, FF)
# save data to file
with open(OUTPREF+'.out', 'w') as output:
    output.write('# -------------------------------------------------------------\n')
    output.write('# parameters\n')
    output.write('# ---------------------------------------------------------------\n')
    output.write('potential:                   %s' % EL.lower())
    output.write('pressure index:              %d' % PI)
    output.write('number of sets:              %d' % NT)
    output.write('number of samples (per set): %d' % NS)
    output.write('training samples (per set):  %d' % LN)
    output.write('feature:                     %s' % FTR)
    output.write('scaler:                      %s' % SCLR)
    output.write('reduction:                   %s' % RDCN)
    output.write('clustering:                  %s' % CLST)
    output.write('fitting function:            %s' % FF)
    output.write('# ------------------------------------------------------------\n')
    output.write('# transition | standard error\n')
    output.write('%f %f\n' % (TRANS, CERR))
    output.write('# liquid probability | standard error\n')
    for i in range(NT):
        output.write('%f %f\n' % (MPRED[i], SPRED[i]))
    output.write('# potential | standard error\n')
    for i in range(NT):
        output.write('%f %f\n' % (np.mean(PE[i, :]), np.std(PE[i, :])))
    output.write('# virial | standard error\n')
    for i in range(NT):
        output.write('%f %f\n' % (np.mean(VIRIAL[i, :]), np.std(VIRIAL[i, :])))
    output.write('# temperature | standard error\n')
    for i in range(NT):
        output.write('%f %f\n' % (np.mean(TEMP[i, :]), np.std(TEMP[i, :])))


def plot_reduction():
    ''' plot of reduced sample space '''
    fig = plt.figure()
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 2),
                     axes_pad=2.0,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="4%",
                     cbar_pad=0.4)
    for i in range(len(grid)):
        grid[i].spines['right'].set_visible(False)
        grid[i].spines['top'].set_visible(False)
        grid[i].xaxis.set_ticks_position('bottom')
        grid[i].yaxis.set_ticks_position('left')
    cbd = grid[0].scatter(RDATA[:, 0], RDATA[:, 1], c=TEMP.reshape(NT*LN),
                          cmap=CM, s=120, alpha=0.05, edgecolors='none')
    grid[0].set_aspect('equal', 'datalim')
    grid[0].set_xlabel(r'$x_0$')
    grid[0].set_ylabel(r'$x_1$')
    grid[0].set_title(r'$\mathrm{(a)\enspace Sample\enspace Temperature}$', y=1.02)
    for j in range(2):
        grid[1].scatter(RDATA[PRED == j, 0], RDATA[PRED == j, 1],
                        c=CM(SCALE(CMT[j])), s=120, alpha=0.05, edgecolors='none')
    grid[1].set_aspect('equal', 'datalim')
    grid[1].set_xlabel(r'$x_0$')
    grid[1].set_ylabel(r'$x_1$')
    grid[1].set_title(r'$\mathrm{(b)\enspace Cluster\enspace Temperature}$', y=1.02)
    cbar = grid[0].cax.colorbar(cbd)
    cbar.solids.set(alpha=1)
    grid[0].cax.toggle_label(True)
    fig.savefig(OUTPREF+'.red.png')


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
        ax.scatter(TEMP.reshape(NT*LN)[PRED == j], PRED[PRED == j], c=CM(SCALE(CMT[j])),
                   s=120, alpha=0.05, edgecolors='none')
    ax.scatter(TDOM, MPRED, color=CM(SCALE(TDOM)), s=240, edgecolors='none', marker='*')
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
    ''' plot of classified features '''
    labels = ['Solid', 'Liquid']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    for j in range(2):
        ax.plot(FDOM[FTR], np.mean(FTRS[FTR].reshape(NT*LN, -1)[PRED == j], axis=0),
                color=CM(SCALE(CMT[j])), alpha=1.00, label=labels[0])
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
    plot_reduction()
    plot_phase_probs()
    plot_ftrs()
    if VERBOSE:
        print('plots saved')
        print('------------------------------------------------------------')
if VERBOSE:
    print('finished')
    print('------------------------------------------------------------')