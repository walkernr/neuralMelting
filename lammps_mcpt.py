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
n_smpl = 1536         # number of samples
mod = 128             # frequency of data storage
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
dbox = 4*0.0009765625*lat[el][1]*np.ones((n_press, n_temp))
# max pos adjustment
dpos = 4*0.0009765625*lat[el][1]*np.ones((n_press, n_temp))  
# timestep
timestep = {'real': 4.0,
            'metal': 0.00390625,
            'lj': 0.00390625}
dt = timestep[units[el]]*np.ones((n_press, n_temp))
      
# ----------------
# unit definitions
# ----------------

def define_constants(units, P, T):
    ''' sets thermodynamic constants according to chosen unit system '''
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
    ''' returns file prefix for simulation '''
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
    ''' initializes lammps object and data storage files '''
    # generate input file
    lmpsfilein = lammps_input(el, units, lat, sz, mass, P, dt, lj_param)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfilein)
    # open data storage files
    thermo = open(lmpsfilein.replace('.in', '%02d%02d.thrm' % (i, j)), 'w')
    traj = open(lmpsfilein.replace('.in', '%02d%02d.traj' % (i, j)), 'w')
    # return lammps object and data storage files
    return lmps, thermo, traj
    
# -----------------------------
# output file utility functions
# -----------------------------
    
def thermo_header(thermo, n_smpl, mod, n_swps, ppos, pvol, phmc, n_stps, seed,
                  el, units, lat, sz, mass, P, T, dt, dpos, dbox):
    ''' writes header containing simulation information to thermo file '''
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
    ''' writes thermodynamic properties to thermo file '''
    # extract physical properties
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
    ''' writes trajectory data to traj file '''
    # extract physical properties
    natoms = lmps.extract_global('natoms', 0)
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    # write data to file
    traj.write('%d %.4E\n' % (natoms, box))
    for k in xrange(natoms):
        traj.write('%.4E %.4E %.4E\n' % tuple(x[3*k:3*k+3]))

# -----------------
# monte carlo moves
# -----------------

def position_mc(lmps, Et, ntrypos, naccpos, dpos):
    ''' classic position monte carlo 
        loops through nudging atoms
        accepts/rejects based on energy metropolis criterion ''' 
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
    # return lammps object and tries/acceptations
    return lmps, ntrypos, naccpos
    
def volume_mc(lmps, Et, Pf, ntryvol, naccvol, dbox):
    ''' isobaric-isothermal volume monte carlo
        scales box and positions
        accepts/rejects based on enthalpy metropolis criterion '''
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
    # return lammps object and tries/acceptations
    return lmps, ntryvol, naccvol
    
def hamiltonian_mc(lmps, Et, ntryhmc, nacchmc, T, dt):
    ''' hamiltionian monte carlo
        short md run at generated velocities for desired temp
        accepts/rejects based on energy metropolis criterion '''
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
    # return lammps object and tries/acceptations
    return lmps, ntryhmc, nacchmc
    
# ----------------------------
# monte carlo parameter update
# ----------------------------

def update_mc_param(dpos, dbox, dt, accpos, accvol, acchmc):
    ''' adaptive update of monte carlo parameters '''
    # update position displacment for pos-mc
    if accpos < 0.5:
        dpos *= 0.9375
    else:
        dpos *= 1.0625
    # update box displacement for vol-mc
    if accvol < 0.5:
        dbox *= 0.9375
    else:
        dbox *= 1.0625
    # update timestep for hmc
    if acchmc < 0.5:
        dt *= 0.9375
    else:
        dt *= 1.0625
    # return new mc params
    return dpos, dbox, dt
    
# ------------------
# parallel tempering
# ------------------

def rep_exch(lmps, Et, Pf):
    ''' performs parallel tempering acrros all samples 
        accepts/rejects based on enthalpy metropolis criterion '''
    # simulation set shape
    n_press, n_temp = lmps.shape
    # flatten lammps objects and constants
    lmps = lmps.reshape(-1)
    Et = Et.reshape(-1)
    Pf = Pf.reshape(-1)
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
    print('%d parallel tempering swaps performed: ' % swaps, ' '.join(pairs))
    # return list of lammps objects
    return np.reshape(lmps, (n_press, n_temp))

def rep_exch_bias(lmps, Et, Pf):
    ''' performs parallel tempering acrros all samples 
        accepts/rejects based on enthalpy metropolis criterion '''
    # simulation set shape
    n_press, n_temp = lmps.shape
    n_cfg = n_press*n_temp
    # flatten lammps objects and constants
    lmps = lmps.reshape(-1)
    Et = Et.reshape(-1)
    Pf = Pf.reshape(-1)
    # initialize enthalpy matrix
    ind = np.zeros((n_cfg*(n_cfg-1)/2, 2), dtype=int) 
    dH = np.zeros(n_cfg*(n_cfg-1)/2, dtype=float)
    # loop through upper right triangular matrix
    k = 0
    for i in xrange(n_cfg):
        for j in xrange(i+1, n_cfg):
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
            ind[k] = [i, j]
            dH[k] = (etoti-etotj)*(1/Et[i]-1/Et[j])+(Pf[i]-Pf[j])*(voli-volj)
            k += 1
    # order from largest to greatest by change in enthalphy
    ind = ind[np.argsort(dH)[::-1]]
    dH = np.sort(dH)[::-1]
    candind = np.reshape(ind[0, :], (1, 2))
    # catalog swaps and swapping pairs
    swaps = 0
    pairs = []
    for k in xrange(0, len(ind)):
        if k == 0:
            i, j = ind[0, :]
            candind = np.reshape(ind[0, :], (1, 2))
            if np.random.rand() <= np.min([1, np.exp(dH[k])]):
                swaps += 1
                pairs.append('(%d, %d)' % (i, j))
                # swap lammps objects
                lmps[j], lmps[i] = lmps[i], lmps[j]
        else:
            i, j = ind[k, :]
            if (i not in candind) and (j not in candind):
                candind = np.concatenate((candind, np.reshape(ind[k, :], (1, 2))), axis=0)
                if np.random.rand() <= np.min([1, np.exp(dH[k])]):
                    swaps += 1
                    pairs.append('(%d, %d)' % (i, j))
                    # swap lammps objects
                    lmps[j], lmps[i] = lmps[i], lmps[j]
    print('%d parallel tempering swaps performed: ' % swaps, ' '.join(pairs))
    # return list of lammps objects
    return np.reshape(lmps, (n_press, n_temp))
    
# ---------------------
# monte carlo procedure
# ---------------------

def move_mc(lmps, Et, Pf,
            ppos, pvol, phmc,
            ntrypos, naccpos,
            ntryvol, naccvol,
            ntryhmc, nacchmc, 
            dpos, dbox, T, dt):
    ''' performs monte carlo moves '''
    roll = np.random.rand()
    # position monte carlo
    if roll <= ppos:
        lmps, ntrypos, naccpos = position_mc(lmps, Et, ntrypos, naccpos, dpos)
    # volume monte carlo
    elif roll <= (ppos+pvol):
        lmps, ntryvol, naccvol = volume_mc(lmps, Et, Pf, ntryvol, naccvol, dbox)
    # hamiltonian monte carlo
    else:
        lmps, ntryhmc, nacchmc = hamiltonian_mc(lmps, Et, ntryhmc, nacchmc, T, dt)
    # return lammps object and tries/acceptations counts
    return lmps, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc
    
def get_sample(lmps, Et, Pf,
               ppos, pvol, phmc,
               ntrypos, naccpos,
               ntryvol, naccvol,
               ntryhmc, nacchmc,
               dpos, dbox, T, dt,
               mod, thermo, traj):
    ''' performs enough monte carlo moves to generate a sample (determined by mod) '''
    # loop through monte carlo moves
    for i in xrange(mod):
        dat =  move_mc(lmps, Et, Pf,
                       ppos, pvol, phmc,
                       ntrypos, naccpos,
                       ntryvol, naccvol,
                       ntryhmc, nacchmc,
                       dpos, dbox, T, dt)
        lmps = dat[0]
        ntrypos, naccpos = dat[1:3]
        ntryvol, naccvol = dat[3:5]
        ntryhmc, nacchmc = dat[5:7]
    # acceptation ratios
    accpos = np.nan_to_num(np.float64(naccpos)/np.float64(ntrypos))
    accvol = np.nan_to_num(np.float64(naccvol)/np.float64(ntryvol))
    acchmc = np.nan_to_num(np.float64(nacchmc)/np.float64(ntryhmc))
    # update mc params
    dpos, dbox, dt = update_mc_param(dpos, dbox, dt, accpos, accvol, acchmc)
    # write to data storage files
    write_thermo(thermo, lmps, accpos, accvol, acchmc)
    write_traj(traj, lmps)
    # return lammps object, tries/acceptation counts, and mc params
    return lmps, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, dt
    
def get_samples(lmps, Et, Pf, 
                ppos, pvol, phmc,
                ntrypos, naccpos,
                ntryvol, naccvol,
                ntryhmc, nacchmc,
                dpos, dbox, T, dt,
                mod, thermo, traj):
    ''' performs monte carlo for all configurations to generate new samples '''
    n_press, n_temp = lmps.shape
    # loop through pressures
    for i in xrange(n_press):
        # loop through temperatures
        for j in xrange(n_temp):
            # get new sample configuration for press/temp combo
            dat = get_sample(lmps[i, j], Et[i, j], Pf[i, j],
                             ppos, pvol, phmc,
                             ntrypos[i, j], naccpos[i, j],
                             ntryvol[i, j], naccvol[i, j],
                             ntryhmc[i, j], nacchmc[i, j],
                             dpos[i, j], dbox[i, j], T[j], dt[i, j],
                             mod, thermo[i, j], traj[i, j])
            lmps[i, j] = dat[0]
            ntrypos[i, j], naccpos[i, j] = dat[1:3]
            ntryvol[i, j], naccvol[i, j] = dat[3:5]
            ntryhmc[i, j], nacchmc[i, j] = dat[5:7]
            dpos[i, j], dbox[i, j], dt[i, j] = dat[7:]
    # return lammps object, tries/acceptation counts, and mc params
    return lmps, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, dt

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
ntryvol = np.zeros((n_press, n_temp), dtype=float)
naccvol = np.zeros((n_press, n_temp), dtype=float)
ntryhmc = np.zeros((n_press, n_temp), dtype=float)
nacchmc = np.zeros((n_press, n_temp), dtype=float)
# loop through pressures
for i in xrange(len(P[el])):
    # loop through temperatures
    for j in xrange(n_temp):
        # set thermo constants
        Et[i, j], Pf[i, j] = define_constants(units[el], P[el][i], T[el][j])
        # initialize lammps object and data storage files
        dat = init_lammps(i, j, el, units[el], lat[el], sz[el], mass[el],
                          P[el][i], dt[i, j], lj_param)
        lmps[i, j] = dat[0]
        thermo[i, j], traj[i, j] = dat[1:]
        # write thermo file header
        thermo_header(thermo[i, j], n_smpl, mod, n_swps, ppos, pvol, phmc,
                      n_stps, seed, el, units[el], lat[el], sz[el], mass[el],
                      P[el][i], T[el][j], dt[i, j], dpos[i, j], dbox[i, j])

# -----------                      
# monte carlo
# -----------

# loop through to number of samples that need to be collected
for i in xrange(n_smpl):
    # collect samples for all configurations
    dat = get_samples(lmps, Et, Pf, ppos, pvol, phmc,
                      ntrypos, naccpos, ntryvol,
                      naccvol, ntryhmc, nacchmc,
                      dpos, dbox, T[el], dt,
                      mod, thermo, traj)
    lmps = dat[0]
    ntrypos, naccpos = dat[1:3]
    ntryvol, naccvol = dat[3:5]
    ntryhmc, nacchmc = dat[5:7]
    dpos, dbox, dt = dat[7:]
    # perform replica exchange markov chain monte carlo (parallel tempering)
    lmps = rep_exch(lmps, Et, Pf)

# ----------------------
# close files and lammps
# ----------------------

# loop through pressures
for i in xrange(n_press):
    # loop through temperatures
    for j in xrange(n_temp):
        lmps[i, j].close()
        thermo[i, j].close()
        traj[i, j].close()

# ------------------
# final data storage
# ------------------

# construct data storage file name lists
thnms = [[thermo[i, j].name for j in xrange(n_temp)] for i in xrange(n_press)]
trnms = [[traj[i, j].name for j in xrange(n_temp)] for i in xrange(n_press)]
# loop through pressures
for i in xrange(n_press):
    # get prefix
    prefix = fpref(name, el, lat[el], P[el][i])
    # open collected thermo data file
    with open(prefix+'.thrm', 'w') as fout:
        # open all thermo files
        fin = fileinput.input(thnms[i])
        # write data to collected thermo file
        for line in fin:
            fout.write(line)
        fin.close()
    # open collected traj data file
    with open(prefix+'.traj', 'w') as fout:
        # open traj files
        fin = fileinput.input(trnms[i])
        # write data to collected traj file
        for line in fin:
            fout.write(line)
        fin.close()
# remove old files 
for i in xrange(n_press):
    for j in xrange(n_temp):
        os.remove(thnms[i][j])
        os.remove(trnms[i][j])