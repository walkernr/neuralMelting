# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 20:11:57 2018

@author: Nicholas
"""

from __future__ import division, print_function
import argparse
import os
import pickle
import numpy as np
import numba as nb

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-v', '--verbose', help='verbose output', action='store_true')
PARSER.add_argument('-p', '--parallel', help='parallel run', action='store_true')
PARSER.add_argument('-d', '--distributed', help='distributed run', action='store_true')
PARSER.add_argument('-q', '--queue', help='submission queue',
                    type=str, default='jobqueue')
PARSER.add_argument('-a', '--allocation', help='submission allocation',
                    type=str, default='startup')
PARSER.add_argument('-nn', '--nodes', help='number of nodes',
                    type=int, default=1)
PARSER.add_argument('-np', '--procs_per_node', help='number of processors per node',
                    type=int, default=16)
PARSER.add_argument('-w', '--walltime', help='job walltime',
                    type=int, default=24)
PARSER.add_argument('-m', '--memory', help='total job memory',
                    type=int, default=32)
PARSER.add_argument('-nw', '--workers', help='total job worker count',
                    type=int, default=16)
PARSER.add_argument('-nt', '--threads', help='threads per worker',
                    type=int, default=1)
PARSER.add_argument('-n', '--name', help='name of simulation',
                    type=str, default='test')
PARSER.add_argument('-e', '--element', help='element choice',
                    type=str, default='LJ')
PARSER.add_argument('-i', '--pressure_index', help='pressure index',
                    type=int, default=0)

ARGS = PARSER.parse_args()

VERBOSE = ARGS.verbose
PARALLEL = ARGS.parallel
DISTRIBUTED = ARGS.distributed
QUEUE = ARGS.queue
ALLOC = ARGS.allocation
NODES = ARGS.nodes
PPN = ARGS.procs_per_node
WALLTIME = ARGS.walltime
MEM = ARGS.memory
NWORKER = ARGS.workers
NTHREAD = ARGS.threads
NAME = ARGS.name
EL = ARGS.element
PI = ARGS.pressure_index

if PARALLEL:
    os.environ['DASK_ALLOWED_FAILURES'] = '4'
    from distributed import Client, LocalCluster, progress
    from dask import delayed
if DISTRIBUTED:
    import time
    from dask_jobqueue import PBSCluster

# lattice type
LAT = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}
# file prefix
PREFIX = os.getcwd()+'/'+'%s.%s.%s.%d.lammps' % (NAME, EL.lower(), LAT[EL], PI)


def load_data():
    ''' load atom count, box dimension, and atom positions '''
    # load pickles
    natoms = pickle.load(open('.'.join([PREFIX, 'natoms', 'pickle'])))
    box = pickle.load(open('.'.join([PREFIX, 'box', 'pickle'])))
    pos = pickle.load(open('.'.join([PREFIX, 'pos', 'pickle'])))
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
    # base values for lattice vectors
    b = [-1, 0, 1]
    # generate lattice vectors for ever direction from base
    br = np.array([[b[i], b[j], b[k]] for i in xrange(3) for j in xrange(3) for k in xrange(3)],
                  dtype=np.uint32)
    # create vector for rdf
    gs = np.zeros((len(natoms), bins), dtype=np.float64)
    # reshape position vector
    pos = pos.reshape((len(natoms), natoms[0], -1))
    # return properties
    return natoms, box, pos, br, r, dr, nrho, dni, gs


@nb.njit
def calculate_rdf(j):
    ''' calculate rdf for sample j '''
    g = np.copy(GS[j, :])
    # loop through lattice vectors
    for k in xrange(BR.shape[0]):
        # displacement vector matrix for sample j
        dvm = POS[j, :]-(POS[j, :]+BOX[j]*BR[k].reshape((1, -1))).reshape((-1, 1, 3))
        # vector of displacements between atoms
        d = np.sqrt(np.sum(np.square(dvm), -1))
        # calculate rdf for sample j
        g[1:] += np.histogram(d, R)[0]
    return g

# get spatial properties
NATOMS, BOX, POS, BR, R, DR, NRHO, DNI, GS = calculate_spatial()
# calculate radial distribution for each sample in parallel

if PARALLEL:
    if DISTRIBUTED:
        # construct distributed cluster
        CLUSTER = PBSCluster(queue=QUEUE, project=ALLOC,
                             resource_spec='nodes=%d:ppn=%d' % (NODES, PPN),
                             walltime='%d:00:00' % WALLTIME,
                             processes=NWORKER, cores=NTHREAD*NWORKER, memory=str(MEM)+'GB')
        CLUSTER.start_workers(1)
        # start client with distributed cluster
        CLIENT = Client(CLUSTER)
        while 'processes=0 cores=0' in str(CLIENT.scheduler_info):
            time.sleep(5)
            print(CLIENT.scheduler_info)
    else:
        # construct local cluster
        CLUSTER = LocalCluster(n_workers=NWORKER, threads_per_worker=NTHREAD)
        # start client with local cluster
        CLIENT = Client(CLUSTER)
    OPERATIONS = [delayed(calculate_rdf)(u) for u in xrange(len(NATOMS))]
    FUTURES = CLIENT.compute(OPERATIONS)
    if VERBOSE:
        print('calculating rdfs for %s at pressure %f' % (EL.lower(), P[PI]))
        progress(FUTURES)
        print('\n')
    RESULTS = CLIENT.gather(FUTURES)
    for u in xrange(len(NATOMS)):
        GS[u, :] = RESULTS[u]
    CLIENT.close()
else:
    for u in xrange(len(NATOMS)):
        GS[u, :] = calculate_rdf(u)

# adjust rdf by atom count and atoms contained by shells
G = np.divide(GS, NATOMS[0]*DNI)
# calculate domain for structure factor
Q = 2*np.pi/DR*np.fft.fftfreq(R.size)[1:int(R.size/2)]
# fourier transform of g
PF = DR*np.exp(-complex(0, 1)*Q*R[0])
FTG = -np.imag(PF*np.fft.fft(R[np.newaxis, :]*(G-1))[:, 1:int(R.size/2)])
# structure factor
S = 1+4*np.pi*NRHO[:, np.newaxis]*np.divide(FTG, Q)
with np.errstate(divide='ignore', invalid='ignore'):
    I = np.multiply(np.nan_to_num(np.multiply(G, np.log(G)))-G+1, np.square(R[:]))
if VERBOSE:
    print('calculations finalized')

# pickle data
pickle.dump(NRHO, open(PREFIX+'.nrho.pickle', 'wb'))
pickle.dump(DNI, open(PREFIX+'.dni.pickle', 'wb'))
pickle.dump(R, open(PREFIX+'.r.pickle', 'wb'))
pickle.dump(G, open(PREFIX+'.rdf.pickle', 'wb'))
pickle.dump(Q, open(PREFIX+'.q.pickle', 'wb'))
pickle.dump(S, open(PREFIX+'.sf.pickle', 'wb'))
pickle.dump(I, open(PREFIX+'.ef.pickle', 'wb'))

if VERBOSE:
    print('all properties pickled')
