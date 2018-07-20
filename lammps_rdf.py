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

# boolean for controlling verbosity
if '--verbose' in sys.argv:
    from tqdm import tqdm
    verbose = True
else:
    verbose = False

# boolean for controlling parallel run
if '--serial' in sys.argv:
    parallel = False
else:
    parallel = True
    os.environ['DASK_ALLOWED_FAILURES'] = '4'
    from distributed import Client, LocalCluster, progress
    from dask import delayed
if parallel:
    # boolean for choosing distributed or local cluster
    if '--distributed' in sys.argv:
        distributed = True
        from dask_jobqueue import PBSCluster
        if '--queue' in sys.argv:
            i = sys.argv.index('--queue')
            # #PBS -q jobqueue
            queue = sys.argv[i+1]
        else:
            queue = 'jobqueue'
        if '--allocation' in sys.argv:
            i = sys.argv.index('--allocation')
            # #PBS ex: -A startup
            alloc = sys.argv[i+1]
        else:
            alloc = 'startup'
        if '--resource' in sys.argv:
            i = sys.argv.index('--resource')
            # #PBS -l nodes=1:ppn=16
            resource = sys.argv[i+1]
        else:
            resource = 'nodes=1:ppn=16'
        if '--walltime' in sys.argv:
            i = sys.argv.index('--walltime')
            # #PBS -l walltime=24:00:00
            walltime = sys.argv[i+1]
        else:
            walltime = '24:00:00'
    else:
        distributed = False 
    # boolean for choosing whether to use multithreading
    if '--threading' in sys.argv:
        processes = False
    else:
        processes = True
    # number of processors
    if '--nworker' in sys.argv:
        i = sys.argv.index('--nworker')
        nworker = int(sys.argv[i+1])
    else:
        if processes:
            nworker = 16
        else:
            nworker = 1
    # threads per worker
    if '--nthread' in sys.argv:
        i = sys.argv.index('--nthread')
        nthread = int(sys.argv[i+1])
    else:
        if processes:
            nthread = 1
        else:
            nthread = 16
    
# simulation name
if '--name' in sys.argv:
    i = sys.argv.index('--name')
    name = sys.argv[i+1]
else:
    name = 'remcmc'

# element
if '--element' in sys.argv:
    i = sys.argv.index('--element')
    el = sys.argv[i+1]
else:
    el = 'LJ'
    
# number of pressure sets
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
# pressure index
if '--pressindex' in sys.argv:
    i = sys.argv.index('--pressindex')
    pressind = int(sys.argv[i+1])
else:
    pressind = 0

# pressure
P = np.linspace(lpress, hpress, npress, dtype=np.float64)
# lattice type
lat = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}
# file prefix
prefix = '%s.%s.%s.%d.lammps' % (name, el.lower(), lat[el], int(P[pressind]))

def schedInit(system, nproc, path):
    ''' creates scheduler file using dask-mpi binary, network is initialized with mpi4py '''
    # for use on most systems
    if system == 'mpi':
        subprocess.call(['mpirun', '--np', str(nproc), 'dask-mpi', '--scheduler-file', path])
    # for use on cray systems
    if system == 'ap':
        subprocess.call(['aprun', '-n', str(nproc), 'dask-mpi', '--scheduler-file', path])
    return

def loadData():
    ''' load atom count, box dimension, and atom positions '''
    # load pickles
    natoms = pickle.load(open('.'.join([prefix, 'natoms', 'pickle'])))
    box = pickle.load(open('.'.join([prefix, 'box', 'pickle'])))
    pos = pickle.load(open('.'.join([prefix, 'pos', 'pickle'])))
    # return data
    return natoms, box, pos
    
def calculateSpatial():
    ''' calculate spatial properties '''
    # load atom count, box dimensions, and atom positions
    natoms, box, pos = loadData()
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
    R = np.array(R)
    # create vector for rdf
    gs = np.zeros((len(natoms), bins), dtype=float)
    # reshape position vector
    pos = pos.reshape((len(natoms), natoms[0], -1))
    # return properties
    return natoms, box, pos, R, r, dr, nrho, dni, gs
    
@nb.njit
def calculateRDF(box, pos, R, r, gs):
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
natoms, box, pos, R, r, dr, nrho, dni, gs = calculateSpatial()
# calculate radial distribution for each sample in parallel
if parallel:
    operations = [delayed(calculateRDF)(box[j], pos[j, :], R, r, gs[j, :]) for j in xrange(len(natoms))]
    if distributed:
        # construct distributed cluster
        cluster = PBSCluster(queue=queue, project=alloc, resource_spec=resource, walltime=walltime, extra=['-N %s' % name, '-o pbs.%s.out' %name])
        # start client with distributed cluster
        client = Client(cluster)
    else:
        # construct local cluster
        cluster = LocalCluster(n_workers=nworker, threads_per_worker=nthread, processes=processes)
        # start client with local cluster
        client = Client(cluster)
    if verbose:
        # display client information
        print(client.scheduler_info)
        print('calculating data for %s at pressure %f' % (el.lower(), P[pressind]))
        futures = client.compute(operations)
        progress(futures)
        results = client.gather(futures)
        print('assigning rdfs')
        for j in tqdm(xrange(len(natoms))):
            gs[j, :] = results[j]
    else:
        futures = client.compute(operations)
        results = client.gather(futures)
        for j in xrange(len(natoms)):
            gs[j, :] = results[j]
    client.close()
else:
    for j in xrange(len(natoms)):
        gs[j, :] = calculateRDF(box[j], pos[j, :], R, r, gs[j, :])

# adjust rdf by atom count and atoms contained by shells
g = np.divide(gs, natoms[0]*dni)
# calculate domain for structure factor
q = 2*np.pi/dr*np.fft.fftfreq(r.size)[1:int(r.size/2)]
# fourier transform of g
ftg = -np.imag(dr*np.exp(-complex(0, 1)*q*r[0])*np.fft.fft(r[np.newaxis, :]*(g-1))[:, 1:int(r.size/2)])
# structure factor
s = 1+4*np.pi*nrho[:, np.newaxis]*np.divide(ftg, q)
with np.errstate(divide='ignore', invalid='ignore'):
    i = np.multiply(np.nan_to_num(np.multiply(g, np.log(g)))-g+1, np.square(r[:]))

# pickle data
pickle.dump(nrho, open(prefix+'.nrho.pickle', 'wb'))
pickle.dump(dni, open(prefix+'.dni.pickle', 'wb'))
pickle.dump(r, open(prefix+'.r.pickle', 'wb'))
pickle.dump(g, open(prefix+'.rdf.pickle', 'wb'))
pickle.dump(q, open(prefix+'.q.pickle', 'wb'))
pickle.dump(s, open(prefix+'.sf.pickle', 'wb'))
pickle.dump(i, open(prefix+'.ef.pickle', 'wb'))