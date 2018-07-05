# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 20:11:57 2018

@author: Nicholas
"""

from __future__ import division, print_function
import os, sys, pickle
import numpy as np
import numba as nb
from distributed import Client, LocalCluster, progress
from dask import delayed

verbose = False      # boolean for controlling verbosity
if verbose:
    from tqdm import tqdm
    
distributed = False  # boolean for choosing distributed or local cluster
processes = True     # boolean for choosing whether to use processes

system = 'mpi'                        # switch for mpirun or aprun
nworkers = 16                         # number of processors
nthreads = 1                          # threads per worker
path = os.getcwd()+'/scheduler.json'  # path for scheduler file

# number of pressure data sets
n_press = 8
# simulation name
name = 'remcmc'

# element and pressure index choice
if '--element' in sys.argv:
    i = sys.argv.index('--element')
    el = sys.argv[i+1]
else:
    el = 'LJ'
if '--pressure_index' in sys.argv:
    i = sys.argv.index('--pressure_index')
    pressind = int(sys.argv[i+1])
else:
    pressind = 0

# pressure
P = {'Ti': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
     'Al': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
     'Ni': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
     'Cu': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
     'LJ': np.linspace(1.0, 8.0, n_press, dtype=np.float64)}
# lattice type
lat = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}
# file prefix
prefix = '%s.%s.%s.%d.lammps' % (name, el.lower(), lat[el], int(P[el][pressind]))

def sched_init(system, nproc, path):
    ''' creates scheduler file using dask-mpi binary, network is initialized with mpi4py '''
    # for use on most systems
    if system == 'mpi':
        subprocess.call(['mpirun', '--np', str(nproc), 'dask-mpi', '--scheduler-file', path])
    # for use on cray systems
    if system == 'ap':
        subprocess.call(['aprun', '-n', str(nproc), 'dask-mpi', '--scheduler-file', path])
    return

def load_data():
    ''' load atom count, box dimension, and atom positions '''
    # load pickles
    natoms = pickle.load(open('.'.join([prefix, 'natoms', 'pickle'])))
    box = pickle.load(open('.'.join([prefix, 'box', 'pickle'])))
    pos = pickle.load(open('.'.join([prefix, 'pos', 'pickle'])))
    # return data
    return natoms, box, pos
    
def calculate_spatial():
    ''' calculate spatial properties '''
    # load atom count, box dimensions, and atom positions
    natoms, box, pos = load_data()
    # number density of atoms
    nrho = np.divide(natoms, np.power(box, 3))
    # minimum box size in simulation
    l = np.min(box)
    # maximum radius for rdf
    mr = np.sqrt(3)/2 # 1/2
    # bin count for rdf
    bins = 256
    # domain for rdf
    r = np.linspace(1e-16, mr, bins)
    # domain spacing for rdf
    dr = r[1]-r[0]
    # differential volume contained by shell at distance r
    dv = 4*np.pi*np.square(r)*dr
    # scaling for spatial properties
    r = r*l
    dr = dr*l
    dv = dv*l**3
    # differential number of atoms contained by shell at distance r
    dni = np.multiply(nrho[:, np.newaxis], dv[np.newaxis, :])
    # vector for containing lattice vectors
    R = []
    # base values for lattice vectors
    base = [-1, 0, 1]
    # generate lattice vectors for ever direction from base
    for i in xrange(len(base)):
        for j in xrange(len(base)):
            for k in xrange(len(base)):
                R.append(np.array([base[i], base[j], base[k]], dtype=float))
    # create vector for rdf
    gs = np.zeros((len(natoms), bins), dtype=float)
    # reshape position vector
    pos = pos.reshape((len(natoms), natoms[0], -1))
    # return properties
    return natoms, box, pos, R, r, dr, nrho, dni, gs
    
@nb.njit
def calculate_rdf(box, pos, R, r, gs):
    ''' calculate rdf for sample j '''
    # loop through lattice vectors
    for k in xrange(R.shape[0]):
        # displacement vector matrix for sample j
        dvm = pos-(pos+box*R[k].reshape((1, -1))).reshape((-1, 1, 3))
        # vector of displacements between atoms
        d = np.sqrt(np.sum(np.square(dvm), -1))
        # calculate rdf for sample j
        gs[1:] += np.histogram(d, r)[0]
    return gs

# get spatial properties
natoms, box, pos, R, r, dr, nrho, dni, gs = calculate_spatial()
# calculate radial distribution for each sample in parallel
operations = [delayed(calculate_rdf)(box[j], pos[j, :], R, r, gs[j, :]) for j in xrange(len(natoms))]
if distributed:
    # construct scheduler with mpi
    sched_init(system, nworkers, path)
    # start client with scheduler file
    client = Client(scheduler=path)
else:
    # construct local cluster
    cluster = LocalCluster(n_workers=nworkers, threads_per_worker=nthreads, processes=processes)
    # start client with local cluster
    client = Client(cluster)
if verbose:
    # display client information
    print(client)
    print('calculating data for %s at pressure %f' % (el.lower(), P[el][pressind]))
    futures = client.compute(operations)
    progress(futures)
    results = client.gather(futures)
    print('assigning rdfs')
    for j in tqdm(xrange(len(natoms))):
        gs[j, :] = resultss[j]
else:
    futures = client.compute(operations)
    results = client.gather(futures)
    for j in xrange(len(natoms)):
        gs[j, :] = results[j]
client.close()

# adjust rdf by atom count and atoms contained by shells
g = np.divide(gs, natoms[0]*dni)
# calculate domain for structure factor
q = 2*np.pi/dr*np.fft.fftfreq(r.size)[1:int(r.size/2)]
# fourier transform of g
ftg = -np.imag(dr*np.exp(-complex(0, 1)*q*r[0])*np.fft.fft(r[np.newaxis, :]*(g-1))[:, 1:int(r.size/2)])
# structure factor
s = 1+4*np.pi*nrho[:, np.newaxis]*np.divide(ftg, q)
i = np.multiply(np.nan_to_num(np.multiply(g, np.log(g)))-g+1, np.square(r[:]))

# pickle data
pickle.dump(nrho, open(prefix+'.nrho.pickle', 'wb'))
pickle.dump(dni, open(prefix+'.dni.pickle', 'wb'))
pickle.dump(r, open(prefix+'.r.pickle', 'wb'))
pickle.dump(g, open(prefix+'.rdf.pickle', 'wb'))
pickle.dump(q, open(prefix+'.q.pickle', 'wb'))
pickle.dump(s, open(prefix+'.sf.pickle', 'wb'))
pickle.dump(i, open(prefix+'.ef.pickle', 'wb'))