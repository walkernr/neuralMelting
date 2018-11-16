# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:23:12 2018

@author: Nicholas
"""

import argparse
import os
import pickle
import numpy as np

# parse command line (help option generated automatically)
PARSER = argparse.ArgumentParser()
PARSER.add_argument('-v', '--verbose', help='verbose output', action='store_true')
PARSER.add_argument('-n', '--name', help='name of simulation',
                    type=str, default='remcmc_init')
PARSER.add_argument('-e', '--element', help='element choice',
                    type=str, default='LJ')
PARSER.add_argument('-i', '--pressure_index', help='pressure index',
                    type=int, default=0)
# parse arguments
ARGS = PARSER.parse_args()
# verbosity of output
VERBOSE = ARGS.verbose
# simulation name
NAME = ARGS.name
# element choice
EL = ARGS.element
# pressure value
PI = ARGS.pressure_index

# lattice type
LAT = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}
# file prefix
PREFIX = os.getcwd()+'/'+'%s.%s.%s.%02d.lammps' % (NAME, EL.lower(), LAT[EL], PI)

if VERBOSE:
    print('parsing data for %s at pressure index %d' % (EL.lower(), PI))
# parse thermo file
TEMP, PE, KE, VIRIAL, VOL, AP, AV, AH = np.split(np.loadtxt(PREFIX+'.thrm', dtype=np.float32), 8, 1)
TEMP = TEMP[:, 0]
PE = PE[:, 0]
KE = KE[:, 0]
VIRIAL = VIRIAL[:, 0]
VOL = VOL[:, 0]
AP = AP[:, 0]
AV = AV[:, 0]
AH = AH[:, 0]
if VERBOSE:
    print('%d thermodynamic property steps parsed' % len(TEMP))

# parse trajectory file
with open(PREFIX+'.traj', 'r') as traj_in:
    DATA = [line.split() for line in traj_in.readlines()]
    NATOMS, BOX = np.split(np.array([values for values in DATA if len(values) == 2]), 2, 1)
    NATOMS = NATOMS.astype(np.uint16)[:, 0]
    BOX = BOX.astype(np.float32)[:, 0]
    X = [np.array(values).astype(np.float32) for values in DATA if len(values) == 3]
    X = np.concatenate(tuple(X), 0).reshape(NATOMS.size, 3*NATOMS[0])
if VERBOSE:
    print('%d trajectory steps parsed' % len(NATOMS))

# pickle data
pickle.dump(TEMP, open(PREFIX+'.temp.pickle', 'wb'))
pickle.dump(PE, open(PREFIX+'.pe.pickle', 'wb'))
pickle.dump(KE, open(PREFIX+'.ke.pickle', 'wb'))
pickle.dump(VIRIAL, open(PREFIX+'.virial.pickle', 'wb'))
pickle.dump(VOL, open(PREFIX+'.vol.pickle', 'wb'))
pickle.dump(AP, open(PREFIX+'.ap.pickle', 'wb'))
pickle.dump(AV, open(PREFIX+'.av.pickle', 'wb'))
pickle.dump(AH, open(PREFIX+'.ah.pickle', 'wb'))
pickle.dump(NATOMS, open(PREFIX+'.natoms.pickle', 'wb'))
pickle.dump(BOX, open(PREFIX+'.box.pickle', 'wb'))
pickle.dump(X, open(PREFIX+'.pos.pickle', 'wb'))

if VERBOSE:
    print('all properties pickled')
