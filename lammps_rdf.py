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

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
parser.add_argument('-p', '--parallel', help='parallel run', action='store_true')
parser.add_argument('-d', '--distributed', help='distributed run', action='store_true')
parser.add_argument('-q', '--queue', help='submission queue', type=str, default='jobqueue')
parser.add_argument('-a', '--allocation', help='submission allocation', type=str, default='startup')
parser.add_argument('-nn', '--nodes', help='number of nodes', type=int, default=1)
parser.add_argument('-np', '--procs_per_node', help='number of processors per node', type=int, default=16)
parser.add_argument('-w', '--walltime', help='job walltime', type=int, default=24)
parser.add_argument('-m', '--memory', help='total job memory', type=int, default=32)
parser.add_argument('-nw', '--workers', help='total job worker count', type=int, default=4)
parser.add_argument('-nt', '--threads', help='threads per worker', type=int, default=1)
parser.add_argument('-n', '--name', help='name of simulation', type=str, default='test')
parser.add_argument('-e', '--element', help='element choice', type=str, default='LJ')
parser.add_argument('-pn', '--pressure_number', help='number of pressures', type=int, default=4)
parser.add_argument('-pr', '--pressure_range', help='pressure range', type=float, nargs=2, default=[2, 8])
parser.add_argument('-i', '--pressure_index', help='pressure index', type=int, default=0)

args = parser.parse_args()

VERBOSE = args.verbose
PARALLEL = args.parallel
DISTRIBUTED = args.distributed
QUEUE = args.queue
ALLOC = args.allocation
NODES = args.nodes
PPN = args.procs_per_node
WALLTIME = args.walltime
MEM = args.memory
NWORKER = args.workers
NTHREAD = args.threads
NAME = args.name
EL = args.element
NPRESS = args.pressure_number
LPRESS, HPRESS = args.pressure_range
PRESSIND = args.pressure_index

if PARALLEL:
    os.environ['DASK_ALLOWED_FAILURES'] = '4'
    from distributed import Client, LocalCluster, progress
    from dask import delayed
if DISTRIBUTED:
    import time
    from dask_jobqueue import PBSCluster

# pressure
P = np.linspace(LPRESS, HPRESS, NPRESS, dtype=np.float64)
# lattice type
LAT = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}
# file prefix
prefix = os.getcwd()+'/'+'%s.%s.%s.%d.lammps' % (NAME, EL.lower(), LAT[EL], int(P[PRESSIND]))


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
                R.append(np.array([base[i], base[j], base[k]], dtype=np.float64))
    R = np.array(R)
    # create vector for rdf
    gs = np.zeros((len(natoms), bins), dtype=np.float64)
    # reshape position vector
    pos = pos.reshape((len(natoms), natoms[0], -1))
    # return properties
    return natoms, box, pos, R, r, dr, nrho, dni, gs


@nb.njit
def calculateRDF(j):
    ''' calculate rdf for sample j '''
    g = np.copy(gs[j, :])
    # loop through lattice vectors
    for k in xrange(R.shape[0]):
        # displacement vector matrix for sample j
        dvm = pos[j, :]-(pos[j, :]+box[j]*R[k].reshape((1, -1))).reshape((-1, 1, 3))
        # vector of displacements between atoms
        d = np.sqrt(np.sum(np.square(dvm), -1))
        # calculate rdf for sample j
        g[1:] += np.histogram(d, r)[0]
    return g

# get spatial properties
natoms, box, pos, R, r, dr, nrho, dni, gs = calculateSpatial()
# calculate radial distribution for each sample in parallel
if PARALLEL:
    operations = [delayed(calculateRDF)(j) for j in xrange(len(natoms))]
    if DISTRIBUTED:
        # construct distributed cluster
        cluster = PBSCluster(queue=QUEUE, project=ALLOC,
                             resource_spec='nodes=%d:ppn=%d' % (NODES, PPN),
                             walltime='%d:00:00' % WALLTIME,
                             processes=NWORKER, cores=NTHREAD*NWORKER, memory=str(MEM)+'GB')
        cluster.start_workers(1)
        # start client with distributed cluster
        client = Client(cluster)
        while 'processes=0 cores=0' in str(client.scheduler_info):
            time.sleep(5)
            print(client.scheduler_info)
    else:
        # construct local cluster
        cluster = LocalCluster(n_workers=NWORKER, threads_per_worker=NTHREAD)
        # start client with local cluster
        client = Client(cluster)
    futures = client.compute(operations)
    if VERBOSE:
        print('calculating rdfs for %s at pressure %f' % (EL.lower(), P[PRESSIND]))
        progress(futures)
    results = client.gather(futures)
    for j in xrange(len(natoms)):
        gs[j, :] = results[j]
    client.close()
else:
    for j in xrange(len(natoms)):
        gs[j, :] = calculateRDF(j)

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
if VERBOSE:
    print('\ncalculations finalized')

# pickle data
pickle.dump(nrho, open(prefix+'.nrho.pickle', 'wb'))
pickle.dump(dni, open(prefix+'.dni.pickle', 'wb'))
pickle.dump(r, open(prefix+'.r.pickle', 'wb'))
pickle.dump(g, open(prefix+'.rdf.pickle', 'wb'))
pickle.dump(q, open(prefix+'.q.pickle', 'wb'))
pickle.dump(s, open(prefix+'.sf.pickle', 'wb'))
pickle.dump(i, open(prefix+'.ef.pickle', 'wb'))

if VERBOSE:
    print('all properties pickled')
