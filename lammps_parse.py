# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:23:12 2018

@author: Nicholas
"""

from __future__ import division, print_function
import argparse
import os
import pickle
import numpy as np

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-v', '--verbose', help='verbose output', action='store_true')
PARSER.add_argument('-n', '--name', help='name of simulation',
                    type=str, default='test')
PARSER.add_argument('-e', '--element', help='element choice',
                    type=str, default='LJ')
PARSER.add_argument('-pn', '--pressure_number', help='number of pressures',
                    type=int, default=4)
PARSER.add_argument('-pr', '--pressure_range', help='pressure range',
                    type=float, nargs=2, default=[2, 8])
PARSER.add_argument('-i', '--pressure_index', help='pressure index',
                    type=int, default=0)

ARGS = PARSER.parse_args()

VERBOSE = ARGS.verbose
NAME = ARGS.name
EL = ARGS.element
NPRESS = ARGS.pressure_number
LPRESS, hpress = ARGS.pressure_range
PRESSIND = ARGS.pressure_index

# pressure
P = np.linspace(LPRESS, hpress, NPRESS, dtype=np.float32)
# lattice type
LAT = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}
# file prefix
prefix = '%s.%s.%s.%d.lammps' % (NAME, EL.lower(), LAT[EL], int(P[PRESSIND]))
# get full directory
prefix = os.getcwd()+'/'+prefix

if VERBOSE:
    print('parsing data for %s at pressure %f' % (EL.lower(), P[PRESSIND]))
# parse thermo file
with open(prefix+'.thrm', 'rb') as fi:
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
if VERBOSE:
    print('%d thermodynamic property steps parsed' % len(temp))
# parse trajectory file
with open(prefix+'.traj', 'rb') as fi:
    iters = iter(fi)
    natoms = []
    box = []
    pos = []
    for lina in iters:
        dat = lina.split()
        if len(dat) == 2:
            natoms.append(int(dat[0]))
            box.append(float(dat[1]))
            x = np.zeros((natoms[-1], 3), dtype=np.float32)
            for j in xrange(natoms[-1]):
                linb = iters.next()
                x[j, :] = np.array(linb.split()).astype(np.float32)
            pos.append(x.reshape(x.size))
if VERBOSE:
    print('%d trajectory steps parsed' % len(natoms))

# pickle data
pickle.dump(np.array(temp, dtype=np.float32), open(prefix+'.temp.pickle', 'wb'))
pickle.dump(np.array(pe, dtype=np.float32), open(prefix+'.pe.pickle', 'wb'))
pickle.dump(np.array(ke, dtype=np.float32), open(prefix+'.ke.pickle', 'wb'))
pickle.dump(np.array(virial, dtype=np.float32), open(prefix+'.virial.pickle', 'wb'))
pickle.dump(np.array(vol, dtype=np.float32), open(prefix+'.vol.pickle', 'wb'))
pickle.dump(np.array(accpos, dtype=np.float32), open(prefix+'.accpos.pickle', 'wb'))
pickle.dump(np.array(accvol, dtype=np.float32), open(prefix+'.accvol.pickle', 'wb'))
pickle.dump(np.array(acchmc, dtype=np.float32), open(prefix+'.acchmc.pickle', 'wb'))
pickle.dump(np.array(natoms, dtype=np.uint16), open(prefix+'.natoms.pickle', 'wb'))
pickle.dump(np.array(box, dtype=np.float32), open(prefix+'.box.pickle', 'wb'))
pickle.dump(np.array(pos, dtype=np.float32), open(prefix+'.pos.pickle', 'wb'))

if VERBOSE:
    print('all properties pickled')
