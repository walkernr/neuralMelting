# -*- coding: utf-8 -*-
"""
Created on Sat May 26 01:22:43 2018

@author: Nicholas
"""

from __future__ import division, print_function
import sys, pickle
import numpy as np
from multiprocessing import cpu_count
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from TanhScaler import TanhScaler
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

nproc = 2  # cpu_count()
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
property = 'entropic_fingerprint'  # property for classification
n_dat = 64                         # number of datasets
scaler = 'tanh'                    # data scaling method
reduction = 'tsne'                 # reduction method
clust = 'agglomerative'            # clustering method
# summary of input
print('------------------------------------------------------------')
print('input summary')
print('------------------------------------------------------------')
print('potential:       %s' % el.lower())
print('number of sets: ', n_dat)
print('property:       ', property)
print('scaler:         ', scaler)
print('reduction:      ', reduction)
print('clustering:     ', clust)
print('------------------------------------------------------------')
# load domains for rdf and sf
R = pickle.load(open(prefix+'.r.pickle'))[:]
Q = pickle.load(open(prefix+'.q.pickle'))[:]
# load simulation data
N = pickle.load(open(prefix+'.natoms.pickle'))
O = np.concatenate(tuple([i*np.ones(int(len(N)/n_dat), dtype=int) for i in xrange(n_dat)]), 0)
P = pickle.load(open(prefix+'.virial.pickle'))
T = pickle.load(open(prefix+'.temp.pickle'))
G = pickle.load(open(prefix+'.rdf.pickle'))
S = pickle.load(open(prefix+'.sf.pickle'))
# sample space reduction for improving performance
smplspc = np.arange(0, N.size, 2)
N = N[smplspc]
O = O[smplspc]
P = P[smplspc]
T = T[smplspc]
G = G[smplspc]
S = S[smplspc]
I = np.multiply(np.nan_to_num(np.multiply(G, np.log(G)))-G+1, np.square(R))
print('data loaded')
print('------------------------------------------------------------')
# property dictionary
propdom = {'radial_distribution':R, 'entropic_fingerprint':R, 'structure_factor':Q}
properties = {'radial_distribution':G, 'entropic_fingerprint':I, 'structure_factor':S}
# scaler dict
scalers = {'standard':StandardScaler(), 'minmax':MinMaxScaler(), 'robust':RobustScaler(), 'tanh':TanhScaler()}
# reduction dimension
if reduction == 'pca':
    npcacomp = 2
if reduction == 'tsne':
    npcacomp = 64
ntsnecomp = 2
# temperature distribution
tdist = np.histogram(T, n_dat, range=(np.min(T), np.max(T)), density=False)
plxty = int(np.mean(tdist[0]))
tdist = np.histogram(T, n_dat, range=(np.min(T), np.max(T)), density=True)
# reduction initialization
pca = PCA(n_components=npcacomp)
tsne = TSNE(n_jobs=nproc, n_components=ntsnecomp, perplexity=plxty, init='random')
print('scaler and reduction initialized')
print('------------------------------------------------------------')
# clustering initialization
agglom = AgglomerativeClustering(n_clusters=2)
kmeans = KMeans(n_jobs=nproc, n_clusters=2, init='k-means++')
spectral = SpectralClustering(n_jobs=nproc, n_clusters=2)
clustering = {'agglomerative':agglom, 'kmeans':kmeans, 'spectral':spectral}
print('clustering initialized')
print('------------------------------------------------------------')
# data scaling and reduction
sdata = scalers[scaler].fit_transform(properties[property])
pdata = pca.fit_transform(sdata)
evar = pca.explained_variance_ratio_
print('data pca reduced')
print('------------------------------------------------------------')
print('pca fit information')
print('------------------------------------------------------------')
print('principal components:     ', len(evar))
print('explained variances:      ', ', '.join(evar[:3].astype('|S32')), '...')
print('total explained variance: ', np.sum(evar))
print('------------------------------------------------------------')
if reduction == 'pca':
    rdata = pdata
if reduction == 'tsne':
    rdata = tsne.fit_transform(pdata)
    error = tsne.kl_divergence_
    print('TSNE information')
    print('------------------------------------------------------------')
    print('kullback-leibler divergence: ', error)
    print('------------------------------------------------------------')
# clustering prediction
rdata = rdata[:, np.argsort(np.max(rdata, 0)-np.min(rdata, 0))]
pred = clustering[clust].fit_predict(rdata)
if clust == 'kmeans':
    print('kmeans fit information')
    print('------------------------------------------------------------')
    print('intertia: ', clustering[clust].inertia_)
    print('------------------------------------------------------------')
# construct phase temperature distributions
cind = []
ctdist = []
cmtemp = np.zeros(2, dtype=float)
for i in xrange(2):
    cind.append(np.where(pred == i))
    ctdist.append(np.histogram(T[cind[i]], n_dat, range=(np.min(T), np.max(T)), density=True))
    cmtemp[i] = np.mean(T[cind[i]])
pind = np.argsort(cmtemp)
# max solid phase temp and min liquid phase temp 
tmax = np.max(ctdist[pind[0]][1][np.nonzero(ctdist[pind[0]][0])])
tmin = np.min(ctdist[pind[1]][1][np.nonzero(ctdist[pind[1]][0])])
# overlap region indices
sind = np.where(np.logical_and(tmin < ctdist[pind[0]][1], ctdist[pind[0]][1] < tmax))
lind = np.where(np.logical_and(tmin < ctdist[pind[1]][1], ctdist[pind[1]][1] < tmax))
# solid phase truncated distribution
sdom = ctdist[pind[0]][1][sind]
sdist = ctdist[pind[0]][0][sind]/np.sum(ctdist[pind[0]][0][sind])
# solid arithmetic average and error
samtemp = np.sum(np.multiply(sdom, sdist))
sastemp = np.sqrt(np.sum(np.multiply(np.square(sdom-samtemp), sdist)))
# solid geometric average and error
sgmtemp = np.exp(np.sum(np.multiply(sdist, np.log(sdom))))
sgstemp = np.exp(np.sqrt(np.sum(np.multiply(sdist, np.log(sdom/sgmtemp)**2))))
# liquid phase truncated distribution
ldom = ctdist[pind[1]][1][lind]
ldist = ctdist[pind[1]][0][lind]/np.sum(ctdist[pind[1]][0][lind])
# liquid arithmetic average and error
lamtemp = np.sum(np.multiply(ldom, ldist))
lastemp = np.sqrt(np.sum(np.multiply(np.square(ldom-lamtemp), ldist)))
# liquid geometric average and error
lgmtemp = np.exp(np.sum(np.multiply(ldist, np.log(ldom)))/np.sum(ldist))
lgstemp = np.exp(np.sqrt(np.sum(np.multiply(ldist, np.log(ldom/lgmtemp)**2))))
# transition mean and error
amt = np.mean([samtemp, lamtemp])
ast = np.mean([sastemp, lastemp])
gmt = np.sqrt(sgmtemp*lgmtemp)
gst = np.sqrt(sgstemp*lgstemp)
# print results
print('transition ')
print('------------------------------------------------------------')
print('transition region:    [', sdom[0], ', ', sdom[-1], ']')
print('solid arith mean:    ', samtemp, '+/-', sastemp)
print('liquid arith mean:   ', lamtemp, '+/-', lastemp)
print('arithmetic mean:     ', amt, '+/-', ast)
print('arithmetic interval:  [', amt-ast, ', ', amt+ast, ']')
print('solid geo mean:      ', sgmtemp, '+/-', (sgstemp-1)*sgmtemp)
print('liquid geo mean:     ', lgmtemp, '+/-', (lgstemp-1)*lgmtemp)
print('gemoetric mean:      ', gmt, '+/-', (gst-1)*gmt)
print('geometric interval:   [', gmt-(gst-1)*gmt, ', ', gmt+(gst-1)*gmt, ']')
print('------------------------------------------------------------')
# color scale
cm = plt.get_cmap('plasma')
scale = lambda temp: (temp-np.min(T))/np.max(T-np.min(T))
print('colormap and scale defined')
print('------------------------------------------------------------')
# reduction plot
fig0 = plt.figure()
grid0 = ImageGrid(fig0, 111,
                  nrows_ncols=(1, 2),
                  axes_pad=1.8,
                  share_all=True,
                  cbar_location="right",
                  cbar_mode="single",
                  cbar_size="4%",
                  cbar_pad=0.4)
for i in xrange(len(grid0)):
    grid0[i].spines['right'].set_visible(False)
    grid0[i].spines['top'].set_visible(False)
    grid0[i].xaxis.set_ticks_position('bottom')
    grid0[i].yaxis.set_ticks_position('left')
cbd = grid0[0].scatter(rdata[:, 0], rdata[:, 1], c=T, cmap=cm, s=120, alpha=0.05, edgecolors='none')
grid0[0].set_aspect('equal', 'datalim')
grid0[0].set_xlabel('$x_0$')
grid0[0].set_ylabel('$x_1$')
grid0[0].set_title('$\mathrm{(a)\enspace Sample\enspace Temperature}$', y=1.02)
for i in xrange(2):
    grid0[1].scatter(rdata[pred == pind[i], 0], rdata[pred == pind[i], 1], c=cm(scale(cmtemp[pind[i]])), s=120, alpha=0.05, edgecolors='none')
grid0[1].set_aspect('equal', 'datalim')
grid0[1].set_xlabel('$x_0$')
grid0[1].set_ylabel('$x_1$')
grid0[1].set_title('$\mathrm{(b)\enspace Cluster\enspace Temperature}$', y=1.02)
cbar = grid0[0].cax.colorbar(cbd)
cbar.solids.set(alpha=1)
grid0[0].cax.toggle_label(True)
# distribution plot
fig1 = plt.figure()
ax10 = fig1.add_subplot(211)
ax11 = fig1.add_subplot(212)
ax10.fill_between(tdist[1][1:], 0, tdist[0], color=cm(scale(np.mean(T))), alpha=0.25)
ax10.set_xlabel('$T$')
ax10.set_ylabel('$p(T)$')
for i in xrange(2):
    ax11.fill_between(ctdist[pind[i]][1][1:], 0, ctdist[pind[i]][0], color=cm(scale(cmtemp[pind[i]])), alpha=0.25)
ax11.axvline(amt, color=cm(scale(amt)))
# ax11.axvline(gmt, color=cm(.25))
for i in xrange(2):
    ax11.axvline(amt+(-1)**i*ast, color=cm(scale(amt+(-1)**i*ast)), linestyle='--')
    # ax11.axvline(gmt+(-1)**i*(gst-1)*gmt, color=cm(scale(gmt+(-1)**i*(gst-1)*gmt)), linestyle='--')
if el == 'LJ':
    ax11.text(1.25*(amt+ast), 1.25*np.min([np.max(ctdist[pind[i]][0]) for i in xrange(2)]), '$T_{\mathrm{arith}} = %2.2f \pm %2.2f$' % (amt, ast))
    # ax11.text(1.25*(amt+ast), 1.5, '$T_{\mathrm{geo}} = %2.2f \pm %2.2f$' % (gmt, gst))
else:
    ax11.text(1.25*(amt+ast), 1.25*np.min([np.max(ctdist[pind[i]][0]) for i in xrange(2)]), '$T_{\mathrm{arith}} = %4.0f \pm %4.0f$' % (amt, ast))
    # ax11.text(1.25*(amt+ast), 1.5, '$T_{\mathrm{arith}} = %4.0f \pm %4.0f$' % (gmt, gst))
ax11.set_xlabel('$T$')
ax11.set_ylabel('$p(T)$')
# property plot
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
if property == 'radial_distribution':
    ax2.set_xlabel('$\mathrm{Radius}$')
    ax2.set_ylabel('$g(r)$')
if property == 'entropic_fingerprint':
    ax2.set_xlabel('$\mathrm{Radius}$')
    ax2.set_ylabel('$I(r)$')
if property == 'structure_factor':
    ax2.set_xlabel('$\mathrm{Wavenumber}$')
    ax2.set_ylabel('$S(q)$')
for i in xrange(2):
    ax2.plot(propdom[property], np.mean(properties[property][pred == pind[i], :], 0), color=cm(scale(cmtemp[pind[i]])))
if el == 'LJ':
    ax2.legend(['$\mathrm{'+'{:2.2f}'.format(cmtemp[pind[i]])+'K}$' for i in xrange(2)])
else:
    ax2.legend(['$\mathrm{'+'{:4.0f}'.format(cmtemp[pind[i]])+'K}$' for i in xrange(2)])
# save figures
plt_pref = [prefix, property, scaler, reduction, clust]
fig0.savefig('.'.join(plt_pref+['red', 'png']))
fig1.savefig('.'.join(plt_pref+['dist', 'png']))
fig2.savefig('.'.join(plt_pref+['strf', 'png']))
# close plots
plt.close('all')
print('plots saved')
print('------------------------------------------------------------')
print('finished')
print('------------------------------------------------------------')