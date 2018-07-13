# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:23:12 2018

@author: Nicholas
"""

from __future__ import division, print_function
import sys, os, pickle
import numpy as np

# boolean for controlling verbosity
if '--verbose' in sys.argv:
    verbose = True
else:
    verbose = False

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
# get full directory
file = os.getcwd()+'/'+prefix

if verbose:
    print('parsing data for %s at pressure %f' % (el.lower(), P[pressind]))
# parse thermo file
with open(file+'.thrm', 'rb') as fi:
    temp = []
    pe = []
    ke = []
    virial = []
    vol = []
    accpos = []
    accvol = []
    acchmc = []
    iters = iter(fi)
    for lina in iters:
        # ignore header
        if '#' not in lina:
            dat = lina.split()
            temp.append(float(dat[0]))
            pe.append(float(dat[1]))
            ke.append(float(dat[2]))
            virial.append(float(dat[3]))
            vol.append(float(dat[4]))
            accpos.append(float(dat[5]))
            accvol.append(float(dat[6]))
            acchmc.append(float(dat[7]))
    # close file
    fi.close()
if verbose:
    print('%d thermodynamic property steps parsed' % len(temp))
# parse trajectory file
with open(file+'.traj', 'rb') as fi:
    iters = iter(fi)
    natoms = []
    box = []
    pos = []
    for lina in iters:
        dat = lina.split()
        if len(dat) == 2:
            natoms.append(int(dat[0]))
            box.append(float(dat[1]))
            x = np.zeros((natoms[-1], 3), dtype=float)
            for j in xrange(natoms[-1]):
                linb = iters.next()
                x[j, :] = np.array(linb.split()).astype(float)
            pos.append(x.reshape(x.size))
    # close file
    fi.close()
if verbose:
    print('%d trajectory steps parsed' % len(natoms))

# pickle data
pickle.dump(np.array(temp, dtype=float), open(prefix+'.temp.pickle', 'wb'))
pickle.dump(np.array(pe, dtype=float), open(prefix+'.pe.pickle', 'wb'))
pickle.dump(np.array(ke, dtype=float), open(prefix+'.ke.pickle', 'wb'))
pickle.dump(np.array(virial, dtype=float), open(prefix+'.virial.pickle', 'wb'))
pickle.dump(np.array(vol, dtype=float), open(prefix+'.vol.pickle', 'wb'))
pickle.dump(np.array(accpos, dtype=float), open(prefix+'.accpos.pickle', 'wb'))
pickle.dump(np.array(accvol, dtype=float), open(prefix+'.accvol.pickle', 'wb'))
pickle.dump(np.array(acchmc, dtype=float), open(prefix+'.acchmc.pickle', 'wb'))
pickle.dump(np.array(natoms, dtype=int), open(prefix+'.natoms.pickle', 'wb'))
pickle.dump(np.array(box, dtype=float), open(prefix+'.box.pickle', 'wb'))
pickle.dump(np.array(pos, dtype=float), open(prefix+'.pos.pickle', 'wb'))

if verbose:
    print('all properties pickled')