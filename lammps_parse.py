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

prefix = ['%s.%s.%d.lammps.%s' % (el[i].lower(), lat[el[i]][0], sz[el[i]], name)  for i in xrange(len(el))]

def parse_data(i):
    print('parsing data for %s' % el[i].lower())
    file = os.getcwd()+'/'+prefix[i]
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
        fi.close()
    print('%d thermodynamic property steps parsed' % len(stp))
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
        fi.close()
    print('%d trajectory steps parsed' % len(natoms))
    pickle.dump(np.array(stp, dtype=int), open(prefix[i]+'.stp.pickle', 'wb'))
    pickle.dump(np.array(temp, dtype=float), open(prefix[i]+'.temp.pickle', 'wb'))
    pickle.dump(np.array(terr, dtype=float), open(prefix[i]+'.terr.pickle', 'wb'))
    pickle.dump(np.array(pe, dtype=float), open(prefix[i]+'.pe.pickle', 'wb'))
    pickle.dump(np.array(ke, dtype=float), open(prefix[i]+'.ke.pickle', 'wb'))
    pickle.dump(np.array(virial, dtype=float), open(prefix[i]+'.virial.pickle', 'wb'))
    pickle.dump(np.array(vol, dtype=float), open(prefix[i]+'.vol.pickle', 'wb'))
    pickle.dump(np.array(acchmc, dtype=float), open(prefix[i]+'.acchmc.pickle', 'wb'))
    pickle.dump(np.array(accvol, dtype=float), open(prefix[i]+'.accvol.pickle', 'wb'))
    pickle.dump(np.array(mdpehmc, dtype=float), open(prefix[i]+'.mdpehmc.pickle', 'wb'))
    pickle.dump(np.array(mdpevol, dtype=float), open(prefix[i]+'.mdpevol.pickle', 'wb'))
    pickle.dump(np.array(natoms, dtype=int), open(prefix[i]+'.natoms.pickle', 'wb'))
    pickle.dump(np.array(box, dtype=float), open(prefix[i]+'.box.pickle', 'wb'))
    pickle.dump(np.array(pos, dtype=float), open(prefix[i]+'.pos.pickle', 'wb'))
    print('all properties pickled')
                
Parallel(n_jobs=cpu_count(), backend='threading', verbose=4)(delayed(parse_data)(i) for i in xrange(len(prefix)))