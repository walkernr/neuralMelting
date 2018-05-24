# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:32:52 2018

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import pickle

el = ['LJ']
lj_param = (1.0, 1.0)
lat = {'Ti': ('bcc', 2.951),
       'Al': ('fcc', 4.046),
       'Ni': ('fcc', 3.524),
       'Cu': ('fcc', 3.597),
       'LJ': ('fcc', 2**(1/6)*lj_param[1])}  # lattice dict [A]
sz = {'Ti': 4,
      'Al': 3,
      'Ni': 3,
      'Cu': 3,
      'LJ': 3}  # size dict
name = 'hmc'

prefix = ['%s.%s.%d.lammps.%s' % (el[i].lower(), lat[el[i]][0], sz[el[i]], name) for i in xrange(len(el))]

def load_data(i):
    natoms = pickle.load(open('.'.join([prefix[i], 'natoms', 'pickle'])))
    box = pickle.load(open('.'.join([prefix[i], 'box', 'pickle'])))
    pos = pickle.load(open('.'.join([prefix[i], 'pos', 'pickle'])))
    return natoms, box, pos
    
def calculate_spatial(i):
    natoms, box, pos = load_data(i)
    nrho = np.divide(natoms, np.power(box, 3))
    l = np.min(box)
    mr = np.sqrt(3)/2 # 1/2
    bins = 256
    r = np.linspace(1e-16, mr, bins)
    dr = r[1]-r[0]
    dv = 4*np.pi*np.square(r)*dr
    r = r*l
    dr = dr*l
    dv = dv*l**3
    dni = np.multiply(nrho[:, np.newaxis], dv[np.newaxis, :])
    R = []
    base = [-1, 0, 1]
    for i in xrange(len(base)):
        for j in xrange(len(base)):
            for k in xrange(len(base)):
                R.append(np.array([base[i], base[j], base[k]], dtype=float))
    gs = np.zeros((len(natoms), bins), dtype=float)
    rb = np.roll(r, -1)[np.newaxis, np.newaxis, :-1]
    ra = r[np.newaxis, np.newaxis, :-1]
    pos = pos.reshape((len(natoms), natoms[0], -1))
    return natoms, box, pos, R, bins, r, nrho, dni, gs, rb, ra
    
def calculate_rdf(j):
    for k in xrange(len(R)):
        dvm = pos[j, :]-(pos[j, :]+box[j]*R[k][np.newaxis, :])[:, np.newaxis]
        d = np.sqrt(np.sum(np.square(dvm), -1))[:, :, np.newaxis]
        gs[j, 1:] += np.sum((d < rb) & (d > ra), (0, 1))

for i in xrange(1):
    natoms, box, pos, R, bins, r, nrho, dni, gs, rb, ra = calculate_spatial(i)
    Parallel(n_jobs=cpu_count(), backend='threading', verbose=4)(delayed(calculate_rdf)(j) for j in xrange(len(natoms)))
    g = np.divide(gs, natoms[0]*dni)
    dr = r[1]-r[0]
    q = 2*np.pi/dr*np.fft.fftfreq(r.size)[1:int(r.size/2)]
    ftg = -np.imag(dr*np.exp(-complex(0, 1)*q*r[0])*np.fft.fft(r[np.newaxis, :]*(g-1))[:, 1:int(r.size/2)])
    s = 1+4*np.pi*nrho[:, np.newaxis]*np.divide(ftg, q)
    pickle.dump(nrho, open(prefix[i]+'.nrho.pickle', 'wb'))
    pickle.dump(dni, open(prefix[i]+'.dni.pickle', 'wb'))
    pickle.dump(r, open(prefix[i]+'.r.pickle', 'wb'))
    pickle.dump(g, open(prefix[i]+'.rdf.pickle', 'wb'))
    pickle.dump(s, open(prefix[i]+'.sf.pickle', 'wb'))