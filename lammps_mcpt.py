# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 04:20:00 2018

@author: Nicholas
"""

from __future__ import division, print_function
import sys, os
import numpy as np
from lammps import lammps
import fileinput

# --------------
# run parameters
# --------------

# element choice
try:
    el = sys.argv[1]
except:
    el = 'LJ'
# number of data sets
n_temp = 64
# simulation name
name = 'mcpt'
# monte carlo parameters
n_smpl = 64           # number of samples
mod = 64              # frequency of data storage
n_swps = n_smpl*mod   # total mc sweeps
ppos = 0.015625       # probability of pos move
pvol = 0.25           # probability of vol move
phmc = 1-ppos-pvol    # probability of hmc move
n_stps = 8            # md steps during hmc
seed = 256            # random seed
np.random.seed(seed)  # initialize rng
nproc = 2  # cpu_count()

# -------------------
# material properties
# -------------------

# unit system
units = {'Ti': 'metal',
         'Al': 'metal',
         'Ni': 'metal',
         'Cu': 'metal',
         'LJ': 'lj'}
# lennard-jones parameters
lj_param = (1.0, 1.0)
# pressure
P = {'Ti': np.array([2, 4], dtype=np.float64),
     'Al': np.array([2, 4], dtype=np.float64),
     'Ni': np.array([2, 4], dtype=np.float64),
     'Cu': np.array([2, 4], dtype=np.float64),
     'LJ': np.array([2, 4, 8], dtype=np.float64)}
n_press = len(P[el])
# temperature
T = {'Ti': np.linspace(256, 2560, n_temp, dtype=np.float64),
     'Al': np.linspace(256, 2560, n_temp, dtype=np.float64),
     'Ni': np.linspace(256, 2560, n_temp, dtype=np.float64),
     'Cu': np.linspace(256, 2560, n_temp, dtype=np.float64),
     'LJ': np.linspace(0.25, 2.5, n_temp, dtype=np.float64)}
# lattice type and parameter
lat = {'Ti': ('bcc', 2.951),
       'Al': ('fcc', 4.046),
       'Ni': ('fcc', 3.524),
       'Cu': ('fcc', 3.615),
       'LJ': ('fcc', 1.122)}
# box size
sz = {'Ti': 5,
      'Al': 3,
      'Ni': 3,
      'Cu': 3,
      'LJ': 4}
# mass
mass = {'Ti': 47.867,
        'Al': 29.982,
        'Ni': 58.693,
        'Cu': 63.546,
        'LJ': 1.0}
# max box adjustment
dbox = 0.0009765625*lat[el][1]*np.ones((n_press, n_temp))
# max pos adjustment
dpos = 0.0009765625*lat[el][1]*np.ones((n_press, n_temp))  
# timestep
timestep = {'real': 4.0,
            'metal': 0.00390625,
            'lj': 0.00390625}
dt = timestep[units[el]]*np.ones((n_press, n_temp))
      
# ----------------
# unit definitions
# ----------------

def define_constants(units, P, T):
    if units == 'real':
        N_A = 6.0221409e23                       # avagadro number [num/mol]
        kB = 3.29983e-27                         # boltzmann constant [kcal/K]
        R = kB*N_A                               # gas constant [kcal/(mol K)]
        Et = R*T                                 # thermal energy [kcal/mol]
        Pf = 1e-30*(1.01325e5*P)/(4.184e3*kB*T)  # metropolis prefactor [1/A^3]
    if units == 'metal':
        kB = 8.61733e-5                          # boltzmann constant [eV/K]
        Et = kB*T                                # thermal energy [eV]
        Pf = 1e-30*(1e5*P)/(1.60218e-19*kB*T)    # metropolis prefactor [1/A^3]
    if units == 'lj':
        kB = 1.0                                 # boltzmann constant (normalized and unitless)
        Et = kB*T                                # thermal energy [T*]
        Pf = P/(kB*T)                            # metropolis prefactor [1/r*^3]
    return Et, Pf

# ---------------------------------
# lammps file/object initialization 
# ---------------------------------

def fpref(name, el, lat, P):
    prefix = '%s.%s.%s.%d.lammps' % (name, el.lower(), lat[0], int(P))
    return prefix

def lammps_input(el, units, lat, sz, mass, P, dt, lj_param=None):
    ''' constructs input file for lammps
        takes element name, lattice definitions, size, and simulation name
        returns input file name '''
    # convert lattice definition list to strings
    prefix = fpref(name, el, lat, P)
    # set lammps file name
    lmpsfilein = prefix+'.in'
    # open lammps file
    lmpsfile = open(lmpsfilein, 'w')
    # file header
    lmpsfile.write('# LAMMPS Monte Carlo: %s\n\n' % el)
    # units and atom style
    lmpsfile.write('units %s\n' % units)
    lmpsfile.write('atom_style atomic\n')
    lmpsfile.write('atom_modify map yes\n\n')
    # construct simulation box
    lmpsfile.write('lattice %s %s\n' % tuple(lat))
    lmpsfile.write('region box block 0 %d 0 %d 0 %d\n' % (3*(sz,)))
    lmpsfile.write('create_box 1 box\n')
    lmpsfile.write('create_atoms 1 box\n\n')
    # potential definitions
    if el == 'Ti':
        lmpsfile.write('pair_style meam/c\n')
        lmpsfile.write('mass 1 47.867\n')
        lmpsfile.write('pair_coeff * * library.meam Ti Al TiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % el)
    if el == 'Al':
        lmpsfile.write('pair_style meam/c\n')
        lmpsfile.write('mass 1 %f\n' % mass)
        lmpsfile.write('pair_coeff * * library.meam Ti Al TiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % el)
    if el == 'Ni':
        lmpsfile.write('pair_style meam/c\n')
        lmpsfile.write('mass 1 %f\n' % mass)
        lmpsfile.write('pair_coeff * * library.Ni.meam Ni Ni.meam %s\n\n' % el)
    if el == 'Cu':
        lmpsfile.write('pair_style meam/c\n')
        lmpsfile.write('mass 1 %f\n' % mass)
        lmpsfile.write('pair_coeff * * library.Cu.meam Cu Cu.meam %s\n\n' % el)
    if el == 'LJ':
        lmpsfile.write('pair_style lj/cut 2.5\n')
        lmpsfile.write('mass 1 %f\n' % mass)
        lmpsfile.write('pair_coeff 1 1 %f %f 2.5\n\n' % lj_param)
    # minimize lattice structure
    lmpsfile.write('fix 1 all box/relax iso %f vmax %f\n' % (P, 0.0009765625))
    lmpsfile.write('minimize 0.0 %f %d %d\n' % (1.49011612e-8, 1024, 8192))
    lmpsfile.write('unfix 1\n')
    # compute kinetic energy
    lmpsfile.write('compute thermo_ke all ke\n\n')
    # initialize
    lmpsfile.write('timestep %f\n' % dt)
    lmpsfile.write('fix 1 all nve\n')
    lmpsfile.write('run 0')
    # close file and return name
    lmpsfile.close()
    return lmpsfilein
    
def init_lammps(i, j, el, units, lat, sz, mass, P, dt, lj_param=None):
    # generate input file
    lmpsfilein = lammps_input(el, units, lat, sz, mass, P, dt, lj_param)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfilein)
    # open data storage files
    thermo = open(lmpsfilein.replace('.in', '%02d%02d.thrm' % (i, j)), 'w')
    traj = open(lmpsfilein.replace('.in', '%02d%02d.traj' % (i, j)), 'w')
    return lmps, thermo, traj
    
# -----------------------------
# output file utility functions
# -----------------------------
    
def thermo_header(thermo, n_smpl, mod, n_swps, ppos, pvol, phmc, n_stps, seed,
                  el, units, lat, sz, mass, P, T, dt, dpos, dbox):
    thermo.write('#----------------------\n')
    thermo.write('# simulation parameters\n')
    thermo.write('#----------------------\n')
    thermo.write('# nsmpl:    %d\n' % n_smpl)
    thermo.write('# mod:      %d\n' % mod)
    thermo.write('# nswps:    %d\n' % n_swps)
    thermo.write('# ppos:     %f\n' % ppos)
    thermo.write('# pvol:     %f\n' % pvol)
    thermo.write('# phmc:     %f\n' % phmc)
    thermo.write('# nstps:    %d\n' % n_stps)
    thermo.write('# seed:     %d\n' % seed)
    thermo.write('#----------------------\n')
    thermo.write('# material properties\n')
    thermo.write('#----------------------\n')
    thermo.write('# element:  %s\n' % el)
    thermo.write('# units:    %s\n' % units)
    thermo.write('# lattice:  %s\n' % lat[0])
    thermo.write('# latpar:   %f\n' % lat[1])
    thermo.write('# size:     %d\n' % sz)
    thermo.write('# mass:     %f\n' % mass)
    thermo.write('# press:    %f\n' % P)
    thermo.write('# temp:     %f\n' % T)
    thermo.write('# dposmax:  %f\n' % dpos)
    thermo.write('# dboxmax:  %f\n' % dbox)
    thermo.write('# timestep: %f\n' % dt)
    thermo.write('# ------------------------------------------------------------\n')
    thermo.write('# | temp | pe | ke | virial | vol | accpos | accvol | acchmc |\n')
    thermo.write('# ------------------------------------------------------------\n')

def write_thermo(thermo, lmps, accpos, accvol, acchmc):
    temp = lmps.extract_compute('thermo_temp', None, 0)
    pe = lmps.extract_compute('thermo_pe', None, 0)
    ke = lmps.extract_compute('thermo_ke', None, 0)
    virial = lmps.extract_compute('thermo_press', None, 0)
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    vol = np.power(box, 3)
    # print thermal argument string
    therm_args = (temp, pe, ke, virial, vol, accpos, accvol, acchmc)
    print('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E' % therm_args)
    # write data to file
    thermo.write('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n' % therm_args)

def write_traj(traj, lmps):
    natoms = lmps.extract_global('natoms', 0)
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    traj.write('%d %.4E\n' % (natoms, box))
    for k in xrange(natoms):
        traj.write('%.4E %.4E %.4E\n' % tuple(x[3*k:3*k+3]))

# -----------------
# monte carlo moves
# -----------------

def position_mc(lmps, Et, ntrypos, naccpos, dpos):
    # get number of atoms
    natoms = lmps.extract_global('natoms', 0)
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    # loop through atoms
    for k in xrange(natoms):
        # update position tries
        ntrypos += 1
        # save current physical properties
        x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
        pe = lmps.extract_compute('thermo_pe', None, 0)/Et
        xnew = np.copy(x)
        xnew[3*k:3*k+3] += (np.random.rand(3)-0.5)*dpos
        xnew[3*k:3*k+3] -= np.floor(xnew[3*k:3*k+3]/box)*box
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
        lmps.command('run 0')
        penew = lmps.extract_compute('thermo_pe', None, 0)/Et
        dE = penew-pe
        if np.random.rand() <= np.min([1, np.exp(-dE)]):
            # update pos acceptations
            naccpos += 1
            # save new physical properties
            x = xnew
            pe = penew
        else:
            # revert physical properties
            lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
            lmps.command('run 0')
    return lmps, ntrypos, naccpos
    
def volume_mc(lmps, Et, Pf, ntryvol, naccvol, dbox):
    # update volume tries
    ntryvol += 1
    # save current physical properties
    natoms = lmps.extract_global('natoms', 0)
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    vol = np.power(box, 3)
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    pe = lmps.extract_compute('thermo_pe', None, 0)/Et
    # save new physical properties
    boxnew = box+(np.random.rand()-0.5)*dbox
    volnew = np.power(boxnew, 3)
    scalef = boxnew/box
    xnew = scalef*x
    # apply new physical properties
    lmps.command('change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box' % (3*(boxnew,)))
    lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
    lmps.command('run 0')
    penew = lmps.extract_compute('thermo_pe', None, 0)/Et
    # calculate enthalpy criterion
    dH = (penew-pe)+Pf*(volnew-vol)-natoms*np.log(volnew/vol)
    if np.random.rand() <= np.min([1, np.exp(-dH)]):
        # update volume acceptations
        naccvol += 1
        # save new physical properties
        box = boxnew
        vol = volnew
        x = xnew
        pe = penew
    else:
        # revert physical properties
        lmps.command('change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box' % (3*(box,)))
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
        lmps.command('run 0')
    return lmps, ntryvol, naccvol
    
def hamiltonian_mc(lmps, Et, ntryhmc, nacchmc, T, dt):
    # update hmc tries
    ntryhmc += 1
    # set new atom velocities and initialize
    lmps.command('velocity all create %f %d dist gaussian' % (T, np.random.randint(1, 2**16)))
    lmps.command('velocity all zero linear')
    lmps.command('velocity all zero angular')
    lmps.command('timestep %f' % dt)
    lmps.command('run 0')
    # save current physical properties
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    v = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3)))
    pe = lmps.extract_compute('thermo_pe', None, 0)/Et
    ke = lmps.extract_compute('thermo_ke', None, 0)/Et
    etot = pe+ke
    # run md
    lmps.command('run %d' % n_stps)  # this part should be implemented as parallel
    # set new physical properties
    xnew = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    vnew = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3)))
    penew = lmps.extract_compute('thermo_pe', None, 0)/Et
    kenew = lmps.extract_compute('thermo_ke', None, 0)/Et
    etotnew = penew+kenew
    # calculate hamiltonian criterion
    dE = etotnew-etot
    # print(penew-pe, kenew-ke)
    if np.random.rand() <= np.min([1, np.exp(-dE)]):
        # update hamiltonian acceptations
        nacchmc += 1
        # save new physical properties
        x = xnew
        v = vnew
        pe = penew
        ke = kenew
    else:
        # revert physical properties
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
        lmps.scatter_atoms('v', 1, 3, np.ctypeslib.as_ctypes(v))
        lmps.command('run 0')
    return lmps, ntryhmc, nacchmc
    
# ----------------------------
# monte carlo parameter update
# ----------------------------

def update_mc_param(dpos, dbox, dt, accpos, accvol, acchmc):
    if accpos < 0.5:
        dpos *= 0.9375
    else:
        dpos *= 1.0625
    if accvol < 0.5:
        dbox *= 0.9375
    else:
        dbox *= 1.0625
    if acchmc < 0.5:
        dt *= 0.9375
    else:
        dt *= 1.0625
    return dpos, dbox, dt
    
# ------------------
# parallel tempering
# ------------------

def par_tempering_swap(lmps, Et, Pf):
    n_press, n_temp = lmps.shape
    lmps = lmps.reshape(-1)
    Et = Et.reshape(-1)
    Pf = Pf.reshape(-1)
    dH = 256*np.ones((len(Et), len(Et)))
    for i in xrange(len(lmps)):
        for j in xrange(i, len(lmps)):
            pei = lmps[i].extract_compute('thermo_pe', None, 0)
            kei = lmps[i].extract_compute('thermo_ke', None, 0)
            etoti = pei+kei
            boxmini = lmps[i].extract_global('boxlo', 1)
            boxmaxi = lmps[i].extract_global('boxhi', 1)
            boxi = boxmaxi-boxmini
            voli = np.power(boxi, 3)
            pej = lmps[j].extract_compute('thermo_pe', None, 0)
            kej = lmps[j].extract_compute('thermo_ke', None, 0)
            etotj = pej+kej
            boxminj = lmps[j].extract_global('boxlo', 1)
            boxmaxj = lmps[j].extract_global('boxhi', 1)
            boxj = boxmaxj-boxminj
            volj = np.power(boxj, 3)
            dH[i, j] = (etoti-etotj)*(1/Et[i]-1/Et[j])+(Pf[i]-Pf[j])*(voli-volj)
    minind = np.dstack(np.unravel_index(np.argsort(dH.ravel()), (len(Et), len(Et))))[0]
    possind = np.reshape(minind[0, :], (1, 2))
    for i in xrange(1, len(Et)):
        if minind[i, 0] not in possind and minind[i, 1] not in possind:
            possind = np.concatenate((possind, np.reshape(minind[i, :], (1, 2))), axis=0)
    swaps = 0
    pairs = []
    for k in xrange(possind.shape[1]):
        i, j = tuple(possind[k, :])
        if np.random.rand() <= np.min([1, np.exp(-dH[i, j])]):
            swaps += 1
            pairs.append('(%d, %d)' % (i, j))
            lmps[j], lmps[i] = lmps[i], lmps[j]
    print('%d parallel tempering swaps performed: ' % swaps, ' '.join(pairs))
    return np.reshape(lmps, (n_press, n_temp))
    
# ---------------------
# monte carlo procedure
# ---------------------

def move_mc(lmps, Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, T, dt):
    roll = np.random.rand()
    if roll <= ppos:
        lmps, ntrypos, naccpos = position_mc(lmps, Et, ntrypos, naccpos, dpos)
    elif roll <= (ppos+pvol):
        lmps, ntryvol, naccvol = volume_mc(lmps, Et, Pf, ntryvol, naccvol, dbox)
    else:
        lmps, ntryhmc, nacchmc = hamiltonian_mc(lmps, Et, ntryhmc, nacchmc, T, dt)
    return lmps, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc
    
def get_sample(lmps, Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, T, dt, mod, thermo, traj):
    for i in xrange(mod):
        dat =  move_mc(lmps, Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, T, dt)
        lmps, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc = dat
    # acceptance ratios
    accpos = np.nan_to_num(np.float64(naccpos)/np.float64(ntrypos))
    accvol = np.nan_to_num(np.float64(naccvol)/np.float64(ntryvol))
    acchmc = np.nan_to_num(np.float64(nacchmc)/np.float64(ntryhmc))
    dpos, dbox, dt = update_mc_param(dpos, dbox, dt, accpos, accvol, acchmc)
    write_thermo(thermo, lmps, accpos, accvol, acchmc)
    write_traj(traj, lmps)
    return lmps, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, dt
    
def get_samples(lmps, Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, T, dt, mod, thermo, traj):
    for i in xrange(lmps.shape[0]):
        for j in xrange(lmps.shape[1]):
            dat = get_sample(lmps[i, j], Et[i, j], Pf[i, j], ppos, pvol, phmc,
                             ntrypos[i, j], naccpos[i, j], ntryvol[i, j], naccvol[i, j], ntryhmc[i, j], nacchmc[i, j],
                             dpos[i, j], dbox[i, j], T[j], dt[i, j], mod, thermo[i, j], traj[i, j])
            lmps[i, j] = dat[0]
            ntrypos[i, j], naccpos[i, j] = dat[1:3]
            ntryvol[i, j], naccvol[i, j] = dat[3:5]
            ntryhmc[i, j], nacchmc[i, j] = dat[5:7]
            dpos[i, j], dbox[i, j], dt[i, j] = dat[7:]

# -----------
# build lists
# -----------

Et = np.zeros((n_press, n_temp), dtype=float)
Pf = np.zeros((n_press, n_temp), dtype=float)
lmps = np.empty((n_press, n_temp), dtype=object)
thermo = np.empty((n_press, n_temp), dtype=object)
traj = np.empty((n_press, n_temp), dtype=object)
ntrypos = np.zeros((n_press, n_temp), dtype=float)
naccpos = np.zeros((n_press, n_temp), dtype=float)
ntryvol = np.zeros((n_press, n_temp), dtype=float)
naccvol = np.zeros((n_press, n_temp), dtype=float)
ntryhmc = np.zeros((n_press, n_temp), dtype=float)
nacchmc = np.zeros((n_press, n_temp), dtype=float)
for i in xrange(len(P[el])):
    for j in xrange(n_temp):
        Et[i, j], Pf[i, j] = define_constants(units[el], P[el][i], T[el][j])
        lmps[i, j], thermo[i, j], traj[i, j] = init_lammps(i, j, el, units[el], lat[el], sz[el], mass[el], P[el][i], dt[i, j], lj_param)
        thermo_header(thermo[i, j], n_smpl, mod, n_swps, ppos, pvol, phmc, n_stps, seed, el, units[el], lat[el], sz[el], mass[el], 
                      P[el][i], T[el][j], dt[i, j], dpos[i, j], dbox[i, j])

# -----------                      
# monte carlo
# -----------

for i in xrange(n_smpl):
    get_samples(lmps, Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, T[el], dt, mod, thermo, traj)
    lmps = par_tempering_swap(lmps, Et, Pf)

# ----------------------
# close files and lammps
# ----------------------

for i in xrange(n_press):
    for j in xrange(n_temp):
        lmps[i, j].close()
        thermo[i, j].close()
        traj[i, j].close()

# -----------------
# output file names
# -----------------

thermo_names = [[thermo[i, j].name for j in xrange(n_temp)] for i in xrange(n_press)]
traj_names = [[traj[i, j].name for j in xrange(n_temp)] for i in xrange(n_press)]

for i in xrange(n_press):
    prefix = fpref(name, el, lat[el], P[el][i])
    with open(prefix+'.thrm', 'w') as fout:
        fin = fileinput.input(thermo_names[i])
        for line in fin:
            fout.write(line)
        fin.close()
    with open(prefix+'.traj', 'w') as fout:
        fin = fileinput.input(traj_names[i])
        for line in fin:
            fout.write(line)
        fin.close()
        
for i in xrange(n_press):
    for j in xrange(n_temp):
        os.remove(thermo_names[i][j])
        os.remove(traj_names[i][j])