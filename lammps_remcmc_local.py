# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:37:03 2018

@author: Nicholas
"""

from __future__ import division, print_function
import sys, os
import numpy as np
from lammps import lammps
from multiprocessing import cpu_count
from joblib import Parallel, delayed

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
name = 'remcmc_test_local'
# monte carlo parameters
cutoff = 512          # sample cutoff
n_smpl = cutoff+1024  # number of samples
mod = 128             # frequency of data storage
n_swps = n_smpl*mod   # total mc sweeps
ppos = 0.015625       # probability of pos move
pvol = 0.25           # probability of vol move
phmc = 1-ppos-pvol    # probability of hmc move
n_stps = 8            # md steps during hmc
seed = 256            # random seed
np.random.seed(seed)  # initialize rng
parallel = True       # boolean for controlling parallel run
nproc = 4  # cpu_count()

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
P = {'Ti': np.array([2, 4, 8], dtype=np.float64),
     'Al': np.array([2, 4, 8], dtype=np.float64),
     'Ni': np.array([2, 4, 8], dtype=np.float64),
     'Cu': np.array([2, 4, 8], dtype=np.float64),
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
dbox = 0.00390625*lat[el][1]*np.ones((n_press, n_temp))
# max pos adjustment
dpos = 0.00390625*lat[el][1]*np.ones((n_press, n_temp))  
# timestep
timestep = {'real': 4.0,
            'metal': 0.00390625,
            'lj': 0.00390625}
dt = timestep[units[el]]*np.ones((n_press, n_temp))
      
# ----------------
# unit definitions
# ----------------

def define_constants(i, j):
    ''' sets thermodynamic constants according to chosen unit system '''
    if units[el] == 'real':
        N_A = 6.0221409e23                                          # avagadro number [num/mol]
        kB = 3.29983e-27                                            # boltzmann constant [kcal/K]
        R = kB*N_A                                                  # gas constant [kcal/(mol K)]
        Etherm = R*T[el][j]                                         # thermal energy [kcal/mol]
        Pfactor = 1e-30*(1.01325e5*P[el][i])/(4.184e3*kB*T[el][j])  # metropolis pressure prefactor [1/A^3]
    if units[el] == 'metal':
        kB = 8.61733e-5                                             # boltzmann constant [eV/K]
        Etherm = kB*T[el][i]                                        # thermal energy [eV]
        Pfactor = 1e-30*(1e5*P[el][i])/(1.60218e-19*kB*T[el][j])    # metropolis pressure prefactor [1/A^3]
    if units[el] == 'lj':
        kB = 1.0                                                    # boltzmann constant (normalized and unitless)
        Etherm = kB*T[el][j]                                        # thermal energy [T*]
        Pfactor = P[el][i]/(kB*T[el][j])                            # metropolis pressure prefactor [1/r*^3]
    return Etherm, Pfactor

# ---------------------------------
# lammps file/object initialization 
# ---------------------------------

def fpref(i):
    ''' returns file prefix for simulation '''
    prefix = '%s.%s.%s.%d.lammps' % (name, el.lower(), lat[el][0], int(P[el][i]))
    return prefix

def lammps_input(i, j):
    ''' constructs input file for lammps
        takes element name, lattice definitions, size, and simulation name
        returns input file name '''
    # convert lattice definition list to strings
    prefix = fpref(i)
    # set lammps file name
    lmpsfilein = prefix+'.in'
    # open lammps file
    lmpsfile = open(lmpsfilein, 'w')
    # file header
    lmpsfile.write('# LAMMPS Monte Carlo: %s\n\n' % el)
    # units and atom style
    lmpsfile.write('units %s\n' % units[el])
    lmpsfile.write('atom_style atomic\n')
    lmpsfile.write('atom_modify map yes\n\n')
    # construct simulation box
    lmpsfile.write('lattice %s %s\n' % tuple(lat[el]))
    lmpsfile.write('region box block 0 %d 0 %d 0 %d\n' % (3*(sz[el],)))
    lmpsfile.write('create_box 1 box\n')
    lmpsfile.write('create_atoms 1 box\n\n')
    # potential definitions
    if el == 'Ti':
        lmpsfile.write('pair_style meam/c\n')
        lmpsfile.write('mass 1 47.867\n')
        lmpsfile.write('pair_coeff * * library.meam Ti Al TiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % el)
    if el == 'Al':
        lmpsfile.write('pair_style meam/c\n')
        lmpsfile.write('mass 1 %f\n' % mass[el])
        lmpsfile.write('pair_coeff * * library.meam Ti Al TiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % el)
    if el == 'Ni':
        lmpsfile.write('pair_style meam/c\n')
        lmpsfile.write('mass 1 %f\n' % mass[el])
        lmpsfile.write('pair_coeff * * library.Ni.meam Ni Ni.meam %s\n\n' % el)
    if el == 'Cu':
        lmpsfile.write('pair_style meam/c\n')
        lmpsfile.write('mass 1 %f\n' % mass[el])
        lmpsfile.write('pair_coeff * * library.Cu.meam Cu Cu.meam %s\n\n' % el)
    if el == 'LJ':
        lmpsfile.write('pair_style lj/cut 2.5\n')
        lmpsfile.write('mass 1 %f\n' % mass[el])
        lmpsfile.write('pair_coeff 1 1 %f %f 2.5\n\n' % lj_param)
    # minimize lattice structure
    lmpsfile.write('fix 1 all box/relax iso %f vmax %f\n' % (P[el][i], 0.0009765625))
    lmpsfile.write('minimize 0.0 %f %d %d\n' % (1.49011612e-8, 1024, 8192))
    lmpsfile.write('unfix 1\n')
    # compute kinetic energy
    lmpsfile.write('compute thermo_ke all ke\n\n')
    # initialize
    lmpsfile.write('timestep %f\n' % dt[i, j])
    lmpsfile.write('fix 1 all nve\n')
    lmpsfile.write('run 0')
    # close file and return name
    lmpsfile.close()
    return lmpsfilein
    
def init_lammps(i, j):
    ''' initializes lammps object and data storage files '''
    # generate input file
    lmpsfilein = lammps_input(i, j)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfilein)
    # open data storage files
    thermo = open(lmpsfilein.replace('.in', '%02d%02d.thrm' % (i, j)), 'wb')
    traj = open(lmpsfilein.replace('.in', '%02d%02d.traj' % (i, j)), 'wb')
    # return lammps object and data storage files
    return lmps, thermo, traj
    
# -----------------------------
# output file utility functions
# -----------------------------
    
def thermo_header(i, j):
    ''' writes header containing simulation information to thermo file '''
    thermo[i, j].write('#----------------------\n')
    thermo[i, j].write('# simulation parameters\n')
    thermo[i, j].write('#----------------------\n')
    thermo[i, j].write('# nsmpl:    %d\n' % n_smpl)
    thermo[i, j].write('# cutoff:   %d\n' % cutoff)
    thermo[i, j].write('# mod:      %d\n' % mod)
    thermo[i, j].write('# nswps:    %d\n' % n_swps)
    thermo[i, j].write('# ppos:     %f\n' % ppos)
    thermo[i, j].write('# pvol:     %f\n' % pvol)
    thermo[i, j].write('# phmc:     %f\n' % phmc)
    thermo[i, j].write('# nstps:    %d\n' % n_stps)
    thermo[i, j].write('# seed:     %d\n' % seed)
    thermo[i, j].write('#----------------------\n')
    thermo[i, j].write('# material properties\n')
    thermo[i, j].write('#----------------------\n')
    thermo[i, j].write('# element:  %s\n' % el)
    thermo[i, j].write('# units:    %s\n' % units[el])
    thermo[i, j].write('# lattice:  %s\n' % lat[el][0])
    thermo[i, j].write('# latpar:   %f\n' % lat[el][1])
    thermo[i, j].write('# size:     %d\n' % sz[el])
    thermo[i, j].write('# mass:     %f\n' % mass[el])
    thermo[i, j].write('# press:    %f\n' % P[el][i])
    thermo[i, j].write('# temp:     %f\n' % T[el][j])
    thermo[i, j].write('# dposmax:  %f\n' % dpos[i, j])
    thermo[i, j].write('# dboxmax:  %f\n' % dbox[i, j])
    thermo[i, j].write('# timestep: %f\n' % dt[i, j])
    thermo[i, j].write('# ------------------------------------------------------------\n')
    thermo[i, j].write('# | temp | pe | ke | virial | vol | accpos | accvol | acchmc |\n')
    thermo[i, j].write('# ------------------------------------------------------------\n')

def write_thermo(i, j):
    ''' writes thermodynamic properties to thermo file '''
    # extract physical properties
    temp = lmps[i, j].extract_compute('thermo_temp', None, 0)
    pe = lmps[i, j].extract_compute('thermo_pe', None, 0)
    ke = lmps[i, j].extract_compute('thermo_ke', None, 0)
    virial = lmps[i, j].extract_compute('thermo_press', None, 0)
    boxmin = lmps[i, j].extract_global('boxlo', 1)
    boxmax = lmps[i, j].extract_global('boxhi', 1)
    box = boxmax-boxmin
    vol = np.power(box, 3)
    # print thermal argument string
    therm_args = (temp, pe, ke, virial, vol, accpos[i, j], accvol[i, j], acchmc[i, j])
    print('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E' % therm_args)
    # write data to file
    thermo[i, j].write('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n' % therm_args)

def write_traj(i, j):
    ''' writes trajectory data to traj file '''
    # extract physical properties
    natoms = lmps[i, j].extract_global('natoms', 0)
    boxmin = lmps[i, j].extract_global('boxlo', 1)
    boxmax = lmps[i, j].extract_global('boxhi', 1)
    box = boxmax-boxmin
    x = np.copy(np.ctypeslib.as_array(lmps[i, j].gather_atoms('x', 1, 3)))
    # write data to file
    traj[i, j].write('%d %.4E\n' % (natoms, box))
    for k in xrange(natoms):
        traj[i, j].write('%.4E %.4E %.4E\n' % tuple(x[3*k:3*k+3]))

# -----------------
# monte carlo moves
# -----------------

def position_mc(i, j):
    ''' classic position monte carlo 
        loops through nudging atoms
        accepts/rejects based on energy metropolis criterion '''
    global lmps, ntrypos, naccpos
    # get number of atoms
    natoms = lmps[i, j].extract_global('natoms', 0)
    boxmin = lmps[i, j].extract_global('boxlo', 1)
    boxmax = lmps[i, j].extract_global('boxhi', 1)
    box = boxmax-boxmin
    # loop through atoms
    for k in xrange(natoms):
        # update position tries
        ntrypos[i, j] += 1
        # save current physical properties
        x = np.copy(np.ctypeslib.as_array(lmps[i, j].gather_atoms('x', 1, 3)))
        pe = lmps[i, j].extract_compute('thermo_pe', None, 0)/Et[i, j]
        xnew = np.copy(x)
        xnew[3*k:3*k+3] += (np.random.rand(3)-0.5)*dpos[i, j]
        xnew[3*k:3*k+3] -= np.floor(xnew[3*k:3*k+3]/box)*box
        lmps[i, j].scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
        lmps[i, j].command('run 0')
        penew = lmps[i, j].extract_compute('thermo_pe', None, 0)/Et[i, j]
        dE = penew-pe
        if np.random.rand() <= np.min([1, np.exp(-dE)]):
            # update pos acceptations
            naccpos[i, j] += 1
            # save new physical properties
            x = xnew
            pe = penew
        else:
            # revert physical properties
            lmps[i, j].scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
            lmps[i, j].command('run 0')
    
def volume_mc(i, j):
    ''' isobaric-isothermal volume monte carlo
        scales box and positions
        accepts/rejects based on enthalpy metropolis criterion '''
    global lmps, ntryvol, naccvol
    # update volume tries
    ntryvol[i, j] += 1
    # save current physical properties
    natoms = lmps[i, j].extract_global('natoms', 0)
    boxmin = lmps[i, j].extract_global('boxlo', 1)
    boxmax = lmps[i, j].extract_global('boxhi', 1)
    box = boxmax-boxmin
    vol = np.power(box, 3)
    x = np.copy(np.ctypeslib.as_array(lmps[i, j].gather_atoms('x', 1, 3)))
    pe = lmps[i, j].extract_compute('thermo_pe', None, 0)/Et[i, j]
    # save new physical properties
    boxnew = box+(np.random.rand()-0.5)*dbox[i, j]
    volnew = np.power(boxnew, 3)
    scalef = boxnew/box
    xnew = scalef*x
    # apply new physical properties
    lmps[i, j].command('change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box' % (3*(boxnew,)))
    lmps[i, j].scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
    lmps[i, j].command('run 0')
    penew = lmps[i, j].extract_compute('thermo_pe', None, 0)/Et[i, j]
    # calculate enthalpy criterion
    dH = (penew-pe)+Pf[i, j]*(volnew-vol)-natoms*np.log(volnew/vol)
    if np.random.rand() <= np.min([1, np.exp(-dH)]):
        # update volume acceptations
        naccvol[i, j] += 1
        # save new physical properties
        box = boxnew
        vol = volnew
        x = xnew
        pe = penew
    else:
        # revert physical properties
        lmps[i, j].command('change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box' % (3*(box,)))
        lmps[i, j].scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
        lmps[i, j].command('run 0')
    
def hamiltonian_mc(i, j):
    ''' hamiltionian monte carlo
        short md run at generated velocities for desired temp
        accepts/rejects based on energy metropolis criterion '''
    global lmps, ntryhmc, nacchmc
    # update hmc tries
    ntryhmc[i, j] += 1
    # set new atom velocities and initialize
    lmps[i, j].command('velocity all create %f %d dist gaussian' % (T[el][j], np.random.randint(1, 2**16)))
    lmps[i, j].command('velocity all zero linear')
    lmps[i, j].command('velocity all zero angular')
    lmps[i, j].command('timestep %f' % dt[i, j])
    lmps[i, j].command('run 0')
    # save current physical properties
    x = np.copy(np.ctypeslib.as_array(lmps[i, j].gather_atoms('x', 1, 3)))
    v = np.copy(np.ctypeslib.as_array(lmps[i, j].gather_atoms('v', 1, 3)))
    pe = lmps[i, j].extract_compute('thermo_pe', None, 0)/Et[i, j]
    ke = lmps[i, j].extract_compute('thermo_ke', None, 0)/Et[i, j]
    etot = pe+ke
    # run md
    lmps[i, j].command('run %d' % n_stps)  # this part should be implemented as parallel
    # set new physical properties
    xnew = np.copy(np.ctypeslib.as_array(lmps[i, j].gather_atoms('x', 1, 3)))
    vnew = np.copy(np.ctypeslib.as_array(lmps[i, j].gather_atoms('v', 1, 3)))
    penew = lmps[i, j].extract_compute('thermo_pe', None, 0)/Et[i, j]
    kenew = lmps[i, j].extract_compute('thermo_ke', None, 0)/Et[i, j]
    etotnew = penew+kenew
    # calculate hamiltonian criterion
    dE = etotnew-etot
    # print(penew-pe, kenew-ke)
    if np.random.rand() <= np.min([1, np.exp(-dE)]):
        # update hamiltonian acceptations
        nacchmc[i, j] += 1
        # save new physical properties
        x = xnew
        v = vnew
        pe = penew
        ke = kenew
    else:
        # revert physical properties
        lmps[i, j].scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
        lmps[i, j].scatter_atoms('v', 1, 3, np.ctypeslib.as_ctypes(v))
        lmps[i, j].command('run 0')
    
# ----------------------------
# monte carlo parameter update
# ----------------------------

def update_mc_param(i, j):
    ''' adaptive update of monte carlo parameters '''
    global dpos, dbox, dt
    # update position displacment for pos-mc
    if accpos[i, j] < 0.5:
        dpos[i, j] *= 0.9375
    else:
        dpos[i, j] *= 1.0625
    # update box displacement for vol-mc
    if accvol[i, j] < 0.5:
        dbox[i, j] *= 0.9375
    else:
        dbox[i, j] *= 1.0625
    # update timestep for hmc
    if acchmc[i, j] < 0.5:
        dt[i, j] *= 0.9375
    else:
        dt[i, j] *= 1.0625
    
# ---------------------
# monte carlo procedure
# ---------------------

def mc_move(i, j):
    ''' performs monte carlo moves '''
    roll = np.random.rand()
    # position monte carlo
    if roll <= ppos:
        position_mc(i, j)
    # volume monte carlo
    elif roll <= (ppos+pvol):
        volume_mc(i, j)
    # hamiltonian monte carlo
    else:
        hamiltonian_mc(i, j)
    
def generate_sample(i, j):
    ''' performs enough monte carlo moves to generate a sample (determined by mod) '''
    global accpos, accvol, acchmc
    # loop through monte carlo moves
    for k in xrange(mod):
        mc_move(i, j)
    # acceptation ratios
    accpos[i, j] = np.nan_to_num(np.float64(naccpos[i, j])/np.float64(ntrypos[i, j]))
    accvol[i, j] = np.nan_to_num(np.float64(naccvol[i, j])/np.float64(ntryvol[i, j]))
    acchmc[i, j] = np.nan_to_num(np.float64(nacchmc[i, j])/np.float64(ntryhmc[i, j]))
    update_mc_param(i, j)
    
def generate_samples():
    ''' performs monte carlo for all configurations to generate new samples '''
    # loop through pressures
    for i in xrange(n_press):
        # loop through temperatures
        for j in xrange(n_temp):
            # generate new sample configuration for press/temp combo
            generate_sample(i, j)
    # write to data storage files
    for i in xrange(n_press):
        for j in xrange(n_temp):
            write_thermo(i, j)
            write_traj(i, j)
    
def generate_samples_par():
    ''' performs monte carlo in parallel for all configurations to generate new samples '''
    # generate new sample configurations for press/temp combos in parallel
    Parallel(n_jobs=nproc, backend='threading', verbose=4)(delayed(generate_sample)(i, j) for i in xrange(n_press) for j in xrange(n_temp))
    # write to data storage files
    for i in xrange(n_press):
        for j in xrange(n_temp):
            write_thermo(i, j)
            write_traj(i, j)
            
# -----------------------------------------
# replica exchange markov chain monte carlo
# -----------------------------------------

def rep_exch():
    ''' performs parallel tempering acrros all samples 
        accepts/rejects based on enthalpy metropolis criterion '''
    global lmps, Et, Pf
    # flatten lammps objects and constants
    lmps = np.reshape(lmps, -1)
    Et = np.reshape(Et, -1)
    Pf = np.reshape(Pf, -1)
    # catalog swaps and swapping pairs
    swaps = 0
    pairs = []
    # loop through upper right triangular matrix
    for i in xrange(len(lmps)):
        for j in xrange(i+1, len(lmps)):
            # energies for configuration i
            pei = lmps[i].extract_compute('thermo_pe', None, 0)
            kei = lmps[i].extract_compute('thermo_ke', None, 0)
            etoti = pei+kei
            # box dimensions for configuration i
            boxmini = lmps[i].extract_global('boxlo', 1)
            boxmaxi = lmps[i].extract_global('boxhi', 1)
            boxi = boxmaxi-boxmini
            voli = np.power(boxi, 3)
            # energies for configuration j
            pej = lmps[j].extract_compute('thermo_pe', None, 0)
            kej = lmps[j].extract_compute('thermo_ke', None, 0)
            etotj = pej+kej
            # box dimensions for configuration j
            boxminj = lmps[j].extract_global('boxlo', 1)
            boxmaxj = lmps[j].extract_global('boxhi', 1)
            boxj = boxmaxj-boxminj
            volj = np.power(boxj, 3)
            # change in enthalpy
            dH = (etoti-etotj)*(1/Et[i]-1/Et[j])+(Pf[i]-Pf[j])*(voli-volj)
            if np.random.rand() <= np.min([1, np.exp(dH)]):
                swaps += 1
                pairs.append('(%d, %d)' % (i, j))
                # swap lammps objects
                lmps[j], lmps[i] = lmps[i], lmps[j]
    lmps = np.reshape(lmps, (n_press, n_temp))
    Et = np.reshape(Et, (n_press, n_temp))
    Pf = np.reshape(Pf, (n_press, n_temp))
    print('%d replica exchanges performed: ' % swaps, ' '.join(pairs))

# -----------
# build lists
# -----------

# thermo constants
Et = np.zeros((n_press, n_temp), dtype=float)
Pf = np.zeros((n_press, n_temp), dtype=float)
# lammps objects and data storage files
lmps = np.empty((n_press, n_temp), dtype=object)
thermo = np.empty((n_press, n_temp), dtype=object)
traj = np.empty((n_press, n_temp), dtype=object)
# monte carlo tries/acceptations
ntrypos = np.zeros((n_press, n_temp), dtype=float)
naccpos = np.zeros((n_press, n_temp), dtype=float)
accpos = np.zeros((n_press, n_temp), dtype=float)
ntryvol = np.zeros((n_press, n_temp), dtype=float)
naccvol = np.zeros((n_press, n_temp), dtype=float)
accvol = np.zeros((n_press, n_temp), dtype=float)
ntryhmc = np.zeros((n_press, n_temp), dtype=float)
nacchmc = np.zeros((n_press, n_temp), dtype=float)
acchmc = np.zeros((n_press, n_temp), dtype=float)
# loop through pressures
for i in xrange(len(P[el])):
    # loop through temperatures
    for j in xrange(n_temp):
        # set thermo constants
        Et[i, j], Pf[i, j] = define_constants(i, j)
        # initialize lammps object and data storage files
        dat = init_lammps(i, j)
        lmps[i, j] = dat[0]
        thermo[i, j], traj[i, j] = dat[1:]
        # write thermo file header
        thermo_header(i, j)

# -----------                      
# monte carlo
# -----------

# loop through to number of samples that need to be collected
for i in xrange(n_smpl):
    # collect samples for all configurations
    if parallel:
        generate_samples_par()
    else:
        generate_samples()
    # perform replica exchange markov chain monte carlo (parallel tempering)
    rep_exch()

# ------------------
# final data storage
# ------------------

# loop through pressures
for i in xrange(n_press):
    # loop through temperatures
    for j in xrange(n_temp):
        thermo[i, j].close()
        traj[i, j].close()

# construct data storage file name lists
fthrm = [[thermo[i, j].name for j in xrange(n_temp)] for i in xrange(n_press)]
ftraj = [[traj[i, j].name for j in xrange(n_temp)] for i in xrange(n_press)]
# loop through pressures
for i in xrange(n_press):
    # get prefix
    prefix = fpref(i)
    # open collected thermo data file
    with open(prefix+'.thrm', 'w') as fo:
        # write data to collected thermo file
        for j in xrange(n_temp):
            with open(fthrm[i][j], 'r') as fi:
                k = 0
                for line in fi:
                    if '#' in line:
                        fo.write(line)
                    else:
                        k += 1
                        if k > cutoff:
                            fo.write(line)
    # open collected traj data file
    with open(prefix+'.traj', 'w') as fo:
        # write data to collected traj file
        for j in xrange(n_temp):
            natoms = lmps[i, j].extract_global('natoms', 0)
            with open(ftraj[i][j], 'r') as fi:
                k = 0
                for line in fi:
                    k += 1
                    if k > (natoms+1)*cutoff:
                        fo.write(line)
        
# -------------------------------
# clean up files and close lammps
# -------------------------------
 
for i in xrange(n_press):
    for j in xrange(n_temp):
        lmps[i, j].close()
        os.remove(fthrm[i][j])
        os.remove(ftraj[i][j])      