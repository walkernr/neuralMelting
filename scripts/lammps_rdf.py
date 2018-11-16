# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 20:11:57 2018

@author: Nicholas
"""

import argparse
import os
import pickle
import numpy as np
import numba as nb
from tqdm import tqdm

def parse_args():
    ''' parse command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
    parser.add_argument('-p', '--parallel', help='parallel run', action='store_true')
    parser.add_argument('-c', '--client', help='dask client run mode', action='store_true')
    parser.add_argument('-d', '--distributed', help='distributed run', action='store_true')
    parser.add_argument('-q', '--queue', help='submission queue',
                        type=str, default='jobqueue')
    parser.add_argument('-a', '--allocation', help='submission allocation',
                        type=str, default='startup')
    parser.add_argument('-nn', '--nodes', help='number of nodes',
                        type=int, default=1)
    parser.add_argument('-np', '--procs_per_node', help='number of processors per node',
                        type=int, default=16)
    parser.add_argument('-w', '--walltime', help='job walltime',
                        type=int, default=2)
    parser.add_argument('-m', '--memory', help='total job memory',
                        type=int, default=32)
    parser.add_argument('-nw', '--workers', help='total job worker count',
                        type=int, default=16)
    parser.add_argument('-nt', '--threads', help='threads per worker',
                        type=int, default=1)
    parser.add_argument('-mt', '--method', help='parallelization method',
                        type=str, default='fork')
    parser.add_argument('-n', '--name', help='name of simulation',
                        type=str, default='remcmc_init')
    parser.add_argument('-e', '--element', help='element choice',
                        type=str, default='LJ')
    parser.add_argument('-i', '--pressure_index', help='pressure index',
                        type=int, default=0)
    args = parser.parse_args()
    return (args.verbose, args.parallel, args.client, args.distributed,
            args.queue, args.allocation, args.nodes, args.procs_per_node,
            args.walltime, args.memory,
            args.workers, args.threads, args.method,
            args.name, args.element, args.pressure_index)


def client_info():
    ''' print client info '''
    info = str(CLIENT.scheduler_info)
    info = info.replace('<', '').replace('>', '').split()[6:8]
    print('\n%s %s' % tuple(info))


def load_data():
    ''' load atom count, box dimension, and atom positions '''
    # load pickles
    natoms = pickle.load(open('.'.join([PREFIX, 'natoms', 'pickle']), 'rb'))
    box = pickle.load(open('.'.join([PREFIX, 'box', 'pickle']), 'rb'))
    pos = pickle.load(open('.'.join([PREFIX, 'pos', 'pickle']), 'rb'))
    # return data
    return natoms, box, pos


def calculate_spatial():
    ''' calculate spatial properties '''
    # load atom count, box dimensions, and atom positions
    natoms, box, pos = load_data()
    # number of samples
    ns = natoms.size
    # number density of atoms
    nrho = np.divide(natoms, np.power(box, 3))
    # minimum box size in simulation
    l = np.min(box)
    # maximum radius for rdf
    mr = 1/2
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
    br = np.array([[b[i], b[j], b[k]] for i in range(3) for j in range(3) for k in range(3)],
                  dtype=np.int8)
    # create vector for rdf
    gs = np.zeros(bins, dtype=np.float32)
    # reshape position vector
    pos = pos.reshape((ns, natoms[0], -1))
    # return properties
    return ns, natoms, box, pos, br, r, dr, nrho, dni, gs


@nb.njit
def calculate_rdf(natoms, box, pos, gs):
    ''' calculate rdf for sample j '''
    # loop through lattice vectors
    for j in range(BR.shape[0]):
        # displacement vector matrix for sample j
        dvm = pos-(pos+box*BR[j].reshape((1, -1))).reshape((-1, 1, 3))
        # vector of displacements between atoms
        d = np.sqrt(np.sum(np.square(dvm), -1))
        # calculate rdf for sample j
        gs[1:] += np.histogram(d, R)[0]
    return gs/natoms


def calculate_rdfs():
    ''' calculate rdfs for all samples '''
    if VERBOSE:
        print('computing %s %s samples at pressure %d' % (NS, EL.lower(), PI))
    if DASK:
        operations = [delayed(calculate_rdf)(NATOMS[i], BOX[i], POS[i], GS) for i in range(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            progress(futures)
            print('\n')
    elif PARALLEL:
        operations = [delayed(calculate_rdf)(NATOMS[i], BOX[i], POS[i], GS) for i in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        if VERBOSE:
            futures = [calculate_rdf(NATOMS[i], BOX[i], POS[i], GS) for i in tqdm(range(NS))]
        else:
            futures = [calculate_rdf(NATOMS[i], BOX[i], POS[i], GS) for i in range(NS)]
    return futures

if __name__ == '__main__':

    (VERBOSE, PARALLEL, DASK, DISTRIBUTED,
     QUEUE, ALLOC, NODES, PPN,
     WALLTIME, MEM,
     NWORKER, NTHREAD, MTHD,
     NAME, EL, PI) = parse_args()

    # processing or threading
    PROC = (NWORKER != 1)
    # ensure all flags are consistent
    if DISTRIBUTED and not DASK:
        DASK = 1
    if DASK and not PARALLEL:
        PARALLEL = 1

    # lattice type
    LAT = {'Ti': 'bcc',
           'Al': 'fcc',
           'Ni': 'fcc',
           'Cu': 'fcc',
           'LJ': 'fcc'}
    # file prefix
    PREFIX = os.getcwd()+'/'+'%s.%s.%s.%02d.lammps' % (NAME, EL.lower(), LAT[EL], PI)

    if PARALLEL:
        from multiprocessing import freeze_support
    if not DASK:
        from joblib import Parallel, delayed
    if DASK:
        os.environ['DASK_ALLOWED_FAILURES'] = '64'
        os.environ['DASK_WORK_STEALING'] = 'True'
        os.environ['DASK_MULTIPROCESSING_METHOD'] = MTHD
        os.environ['DASK_LOG_FORMAT'] = '\r%(name)s - %(levelname)s - %(message)s'
        from distributed import Client, LocalCluster, progress
        from dask import delayed
    if DISTRIBUTED:
        from dask_jobqueue import PBSCluster
        import time

    # client initialization
    if PARALLEL:
        freeze_support()
        if DASK and not DISTRIBUTED:
            # construct local cluster
            CLUSTER = LocalCluster(n_workers=NWORKER, threads_per_worker=NTHREAD, processes=PROC)
            # start client with local cluster
            CLIENT = Client(CLUSTER)
            # display client information
            if VERBOSE:
                client_info()
        if DASK and DISTRIBUTED:
            # construct distributed cluster
            CLUSTER = PBSCluster(queue=QUEUE, project=ALLOC,
                                 resource_spec='nodes=%d:ppn=%d' % (NODES, PPN),
                                 walltime='%d:00:00' % WALLTIME,
                                 processes=NWORKER, cores=NTHREAD*NWORKER, memory=str(MEM)+'GB',
                                 local_dir=os.getcwd())
            CLUSTER.start_workers(1)
            # start client with distributed cluster
            CLIENT = Client(CLUSTER)
            while 'processes=0 cores=0' in str(CLIENT.scheduler_info):
                time.sleep(5)
                if VERBOSE:
                    client_info()

    # get spatial properties
    NS, NATOMS, BOX, POS, BR, R, DR, NRHO, DNI, GS = calculate_spatial()
    if VERBOSE:
        print('data loaded')
    # calculate radial distributions
    G = calculate_rdfs()
    if DASK:
        G = np.array(CLIENT.gather(G), dtype=np.float32)
        CLIENT.close()
    else:
        G = np.array(G, dtype=np.float32)
    # adjust rdf by atom count and atoms contained by shells
    G = np.divide(G, DNI)
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
