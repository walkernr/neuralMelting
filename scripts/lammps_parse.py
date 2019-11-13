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
# parse arguments
ARGS = PARSER.parse_args()
# verbosity of output
VERBOSE = ARGS.verbose
# simulation name
NAME = ARGS.name
# element choice
EL = ARGS.element

# lattice type
LAT = {'Ti': 'bcc',
       'Al': 'fcc',
       'Ni': 'fcc',
       'Cu': 'fcc',
       'LJ': 'fcc'}
# file prefix
PREFIX = os.getcwd()+'/'+'%s.%s.%s.lammps' % (NAME, EL.lower(), LAT[EL])
P = np.load(PREFIX+'.virial.trgt.npy')
T = np.load(PREFIX+'.temp.trgt.npy')
PN, TN = P.size, T.size

if VERBOSE:
    print('parsing data for %s for %d pressure indices and %d temperature indices' % (EL.lower(), PN, TN))
# parse thermo file
(TEMP, PE, KE, VIRIAL, VOL, DX, DV, DT,
 NTP, NAP, NTV, NAV, NTH, NAH, AP, AV, AH) = np.split(np.loadtxt(PREFIX+'.thrm', dtype=np.float32), 17, 1)
TEMP = TEMP[:, 0].reshape(PN, TN, -1)
PE = PE[:, 0].reshape(PN, TN, -1)
KE = KE[:, 0].reshape(PN, TN, -1)
VIRIAL = VIRIAL[:, 0].reshape(PN, TN, -1)
VOL = VOL[:, 0].reshape(PN, TN, -1)
DX = DX[:, 0].reshape(PN, TN, -1)
DV = DV[:, 0].reshape(PN, TN, -1)
DT = DT[:, 0].reshape(PN, TN, -1)
NTP = NTP[:, 0].reshape(PN, TN, -1)
NAP = NAP[:, 0].reshape(PN, TN, -1)
NTV = NTV[:, 0].reshape(PN, TN, -1)
NAV = NAV[:, 0].reshape(PN, TN, -1)
NTH = NTH[:, 0].reshape(PN, TN, -1)
NAH = NAH[:, 0].reshape(PN, TN, -1)
AP = AP[:, 0].reshape(PN, TN, -1)
AV = AV[:, 0].reshape(PN, TN, -1)
AH = AH[:, 0].reshape(PN, TN, -1)
SN = TEMP.shape[2]
if VERBOSE:
    print('%d thermodynamic property steps parsed' % (PN*TN*SN))

# dump data
np.save(PREFIX+'.temp.npy', TEMP)
np.save(PREFIX+'.pe.npy', PE)
np.save(PREFIX+'.ke.npy', KE)
np.save(PREFIX+'.virial.npy', VIRIAL)
np.save(PREFIX+'.vol.npy', VOL)
np.save(PREFIX+'.dx.npy', DX)
np.save(PREFIX+'.dv.npy', DV)
np.save(PREFIX+'.dt.npy', DT)
np.save(PREFIX+'.ntp.npy', NTP)
np.save(PREFIX+'.nap.npy', NAP)
np.save(PREFIX+'.ntv.npy', NTV)
np.save(PREFIX+'.nav.npy', NAV)
np.save(PREFIX+'.nth.npy', NTH)
np.save(PREFIX+'.nah.npy', NAH)
np.save(PREFIX+'.ap.npy', AP)
np.save(PREFIX+'.av.npy', AV)
np.save(PREFIX+'.ah.npy', AH)
del TEMP, PE, KE, VIRIAL, VOL, DX, DV, NTP, NAP, NTV, NAV, NTH, NAH, AP, AV, AH

# parse trajectory file
with open(PREFIX+'.traj', 'r') as traj_in:
    DATA = [line.split() for line in traj_in.readlines()]
    NATOMS, BOX = np.split(np.array([values for values in DATA if len(values) == 2]), 2, 1)
    NATOMS = NATOMS.astype(np.uint16)[:, 0]
    BOX = BOX.astype(np.float32)[:, 0]
    X = [np.array(values).astype(np.float32) for values in DATA if len(values) == 3]
    X = np.concatenate(tuple(X), 0)
    NATOMS = NATOMS.reshape(PN, TN, -1)
    X = X.reshape(PN, TN, NATOMS.shape[2], NATOMS[0, 0, 0], 3)
if VERBOSE:
    print('%d trajectory steps parsed' % (PN*TN*SN))

# dump data
np.save(PREFIX+'.natoms.npy', NATOMS)
np.save(PREFIX+'.box.npy', BOX)
np.save(PREFIX+'.pos.npy', X)

if VERBOSE:
    print('all properties pickled')
