# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:23:12 2018

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
import os
import pickle
from multiprocessing import cpu_count
from joblib import Parallel, delayed

# element choice
try:
    el = sys.argv[1]
except:
    el = 'LJ'
# lennard-jones parameter
lj_param = (1.0, 1.0)
# lattice dict
lat = {'Ti': ('bcc', 2.951),
       'Al': ('fcc', 4.046),
       'Ni': ('fcc', 3.524),
       'Cu': ('fcc', 3.597),
       'LJ': ('fcc', 2**(1/6)*lj_param[1])}
# size dict
sz = {'Ti': 4,
      'Al': 3,
      'Ni': 3,
      'Cu': 3,
      'LJ': 3}
# simulation name
name = 'hmc'
# file prefix
prefix = '%s.%s.%d.lammps.%s' % (el.lower(), lat[el][0], sz[el], name)
# get full directory
file = os.getcwd()+'/'+prefix
print('parsing data for %s' % el.lower())
# parse thermo file
with open(file+'.thrm', 'r') as fi:
    stp = []
    temp = []
    terr = []
    pe = []
    ke = []
    virial = []
    vol = []
    acchmc = []
    accvol = []
    mdpehmc = []
    mdpevol = []
    iters = iter(fi)
    for lina in iters:
        # ignore header
        if '#' not in lina:
            dat = lina.split()
            stp.append(int(dat[0]))
            temp.append(float(dat[1]))
            terr.append(float(dat[2]))
            pe.append(float(dat[3]))
            ke.append(float(dat[4]))
            virial.append(float(dat[5]))
            vol.append(float(dat[6]))
            acchmc.append(float(dat[7]))
            accvol.append(float(dat[8]))
            mdpehmc.append(float(dat[9]))
            mdpevol.append(float(dat[10]))
    # close file
    fi.close()
print('%d thermodynamic property steps parsed' % len(stp))
# parse trajectory file
with open(file+'.traj', 'r') as fi:
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
print('%d trajectory steps parsed' % len(natoms))
# pickle data
pickle.dump(np.array(stp, dtype=int), open(prefix+'.stp.pickle', 'wb'))
pickle.dump(np.array(temp, dtype=float), open(prefix+'.temp.pickle', 'wb'))
pickle.dump(np.array(terr, dtype=float), open(prefix+'.terr.pickle', 'wb'))
pickle.dump(np.array(pe, dtype=float), open(prefix+'.pe.pickle', 'wb'))
pickle.dump(np.array(ke, dtype=float), open(prefix+'.ke.pickle', 'wb'))
pickle.dump(np.array(virial, dtype=float), open(prefix+'.virial.pickle', 'wb'))
pickle.dump(np.array(vol, dtype=float), open(prefix+'.vol.pickle', 'wb'))
pickle.dump(np.array(acchmc, dtype=float), open(prefix+'.acchmc.pickle', 'wb'))
pickle.dump(np.array(accvol, dtype=float), open(prefix+'.accvol.pickle', 'wb'))
pickle.dump(np.array(mdpehmc, dtype=float), open(prefix+'.mdpehmc.pickle', 'wb'))
pickle.dump(np.array(mdpevol, dtype=float), open(prefix+'.mdpevol.pickle', 'wb'))
pickle.dump(np.array(natoms, dtype=int), open(prefix+'.natoms.pickle', 'wb'))
pickle.dump(np.array(box, dtype=float), open(prefix+'.box.pickle', 'wb'))
pickle.dump(np.array(pos, dtype=float), open(prefix+'.pos.pickle', 'wb'))
print('all properties pickled')