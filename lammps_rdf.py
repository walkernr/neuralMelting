# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:32:52 2018

@author: Nicholas
"""

from __future__ import division, print_function
import sys, pickle
import numpy as np
from multiprocessing import cpu_count
from joblib import Parallel, delayed


# element choice
try:
    el = sys.argv[1]
except:
    el = 'LJ'
# pressure
P = {'Ti': 1.0,
     'Al': 1.0,
     'Ni': 1.0,
     'Cu': 1.0,
     'LJ': 1.0}
# lattice type
lat = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}
# simulation name
name = 'mc'
# file prefix
prefix = '%s.%s.%d.lammps.%s' % (el.lower(), lat[el], int(P[el]), name)

nproc = 2  # cpu_count()

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
    # vectors for counting atoms between distances ra and rb
    rb = np.roll(r, -1)[np.newaxis, np.newaxis, :-1]
    ra = r[np.newaxis, np.newaxis, :-1]
    # reshape position vector
    pos = pos.reshape((len(natoms), natoms[0], -1))
    # return properties
    return natoms, box, pos, R, bins, r, dr, nrho, dni, gs, rb, ra
    
def calculate_rdf(j):
    ''' calculate rdf for sample j '''
    # loop through lattice vectors
    for k in xrange(len(R)):
        # displacement vector matrix for sample j
        dvm = pos[j, :]-(pos[j, :]+box[j]*R[k][np.newaxis, :])[:, np.newaxis]
        # vector of displacements between atoms
        d = np.sqrt(np.sum(np.square(dvm), -1))[:, :, np.newaxis]
        # calculate rdf for sample j
        gs[j, 1:] += np.sum((d < rb) & (d > ra), (0, 1))

# get spatial properties
natoms, box, pos, R, bins, r, dr, nrho, dni, gs, rb, ra = calculate_spatial()
# calculate radial distribution for each sample in parallel
Parallel(n_jobs=nproc, backend='threading', verbose=4)(delayed(calculate_rdf)(j) for j in xrange(len(natoms)))
# adjust rdf by atom count and atoms contained by shells
g = np.divide(gs, natoms[0]*dni)
# calculate domain for structure factor
q = 2*np.pi/dr*np.fft.fftfreq(r.size)[1:int(r.size/2)]
# fourier transform of g
ftg = -np.imag(dr*np.exp(-complex(0, 1)*q*r[0])*np.fft.fft(r[np.newaxis, :]*(g-1))[:, 1:int(r.size/2)])
# structure factor
s = 1+4*np.pi*nrho[:, np.newaxis]*np.divide(ftg, q)
# pickle data
pickle.dump(nrho, open(prefix+'.nrho.pickle', 'wb'))
pickle.dump(dni, open(prefix+'.dni.pickle', 'wb'))
pickle.dump(r, open(prefix+'.r.pickle', 'wb'))
pickle.dump(g, open(prefix+'.rdf.pickle', 'wb'))
pickle.dump(q, open(prefix+'.q.pickle', 'wb'))
pickle.dump(s, open(prefix+'.sf.pickle', 'wb'))