# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 04:20:00 2018

@author: Nicholas
"""

from __future__ import division, print_function
import sys, os, subprocess
import numpy as np
from lammps import lammps
from distributed import Client, LocalCluster, progress
from dask import delayed

# --------------
# run parameters
# --------------

verbose = False       # boolean for controlling verbosity

parallel = True       # boolean for controlling parallel run
distributed = False   # boolean for choosing distributed or local cluster
processes = True      # boolean for choosing whether to use processes

system = 'mpi'                        # switch for mpirun or aprun
nworkers = 16                         # number of processors
nthreads = 1                          # threads per worker
path = os.getcwd()+'/scheduler.json'  # path for scheduler file

# number of data sets
n_press = 4
n_temp = 4
# simulation name
name = 'test'
# monte carlo parameters
cutoff = 4         # sample cutoff
n_smpl = cutoff+4  # number of samples
mod = 4             # frequency of data storage
n_swps = n_smpl*mod   # total mc sweeps
ppos = 0.25           # probability of pos move
pvol = 0.25           # probability of vol move
phmc = 1-ppos-pvol    # probability of hmc move
n_stps = 8            # md steps during hmc
seed = 256            # random seed
np.random.seed(seed)  # initialize rng

# element choice
if '--element' in sys.argv:
    i = sys.argv.index('--element')
    el = sys.argv[i+1]
else:
    el = 'LJ'

# -------------------
# material properties
# -------------------

# unit system
units = {'Ti': 'metal',
         'Al': 'metal',
         'Ni': 'metal',
         'Cu': 'metal',
         'LJ': 'lj'}
# pressure
P = {'Ti': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
     'Al': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
     'Ni': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
     'Cu': np.linspace(1.0, 8.0, n_press, dtype=np.float64),
     'LJ': np.linspace(1.0, 8.0, n_press, dtype=np.float64)}
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
sz = {'Ti': 4,
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
# timestep
timestep = {'real': 4.0,
            'metal': 0.00390625,
            'lj': 0.00390625}
# max box adjustment
dbox = 0.125*lat[el][1]*np.ones((n_press, n_temp))
# max pos adjustment
dpos = 0.125*lat[el][1]*np.ones((n_press, n_temp))  
# hmc timestep
dt = timestep[units[el]]*np.ones((n_press, n_temp))

# ---------------------
# client initialization
# ---------------------

def sched_init(system, nproc, path):
    ''' creates scheduler file using dask-mpi binary, network is initialized with mpi4py '''
    # for use on most systems
    if system == 'mpi':
        subprocess.call(['mpirun', '--np', str(nproc), 'dask-mpi', '--scheduler-file', path])
    # for use on cray systems
    if system == 'ap':
        subprocess.call(['aprun', '-n', str(nproc), 'dask-mpi', '--scheduler-file', path])
    return
      
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

def lammps_input(el, units, lat, sz, mass, P, dt):
    ''' constructs input file for lammps
        takes element name, lattice definitions, size, and simulation name
        returns input file name '''
    lj_param = (1.0, 1.0)
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
    # compute kinetic energy
    lmpsfile.write('compute thermo_ke all ke\n\n')
    # initialize
    lmpsfile.write('timestep %f\n' % dt)
    lmpsfile.write('fix 1 all nve\n')
    lmpsfile.write('run 0')
    # close file and return name
    lmpsfile.close()
    return lmpsfilein
    
def lammps_init(x, v, box, el, units, lat, sz, mass, P, dt):
    ''' initializes system info and data storage files '''
    # generate input file
    lmpsfilein = lammps_input(el, units, lat, sz, mass, P, dt)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfilein)
    # set system info
    lmps.command('change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box' % (3*(box,)))
    lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
    lmps.scatter_atoms('v', 1, 3, np.ctypeslib.as_ctypes(v))
    lmps.command('run 0')
    return lmps
    
def lammps_extract(lmps):
    # extract all system info
    natoms = lmps.extract_global('natoms', 0)
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    v = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3)))
    temp = lmps.extract_compute('thermo_temp', None, 0)
    pe = lmps.extract_compute('thermo_pe', None, 0)
    ke = lmps.extract_compute('thermo_ke', None, 0)
    virial = lmps.extract_compute('thermo_press', None, 0)
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    vol = np.power(box, 3)
    return natoms, x, v, temp, pe, ke, virial, box, vol
    
def sample_init(i, j, el, units, lat, sz, mass, P, dpos, dt):
    ''' initializes system info and data storage files '''
    # generate input file
    lmpsfilein = lammps_input(el, units, lat, sz, mass, P, dt)
    # initialize lammps
    lmps = lammps() # cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfilein)
    # minimize lattice structure
    lmps.command('unfix 1')
    lmps.command('fix 1 all box/relax iso %f vmax %f' % (P, 0.0009765625))
    lmps.command('minimize 0.0 %f %d %d' % (1.49011612e-8, 1024, 8192))
    lmps.command('displace_atoms all random %f %f %f %d' % (3*(dpos,)+(np.random.randint(1, 2**16),)))
    # extract all system info
    natoms, x, v, temp, pe, ke, virial, box, vol = lammps_extract(lmps)
    # open data storage files
    thermo = open(lmpsfilein.replace('.in', '%02d%02d.thrm' % (i, j)), 'w')
    traj = open(lmpsfilein.replace('.in', '%02d%02d.traj' % (i, j)), 'w')
    lmps.close()
    # return system info and data storage files
    return natoms, x, v, temp, pe, ke, virial, box, vol, thermo, traj
    
# -----------------------------
# output file utility functions
# -----------------------------
    
def thermo_header(thermo, n_smpl, cutoff, mod, n_swps, ppos, pvol, phmc, n_stps, 
                  seed, el, units, lat, sz, mass, P, T, dt, dpos, dbox):
    ''' writes header containing simulation information to thermo file '''
    thermo.write('#----------------------\n')
    thermo.write('# simulation parameters\n')
    thermo.write('#----------------------\n')
    thermo.write('# nsmpl:    %d\n' % n_smpl)
    thermo.write('# cutoff:   %d\n' % cutoff)
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

def write_thermo(thermo, temp, pe, ke, virial, vol, accpos, accvol, acchmc, verbose):
    ''' writes thermodynamic properties to thermo file '''
    # print thermal argument string
    therm_args = (temp, pe, ke, virial, vol, accpos, accvol, acchmc)
    if verbose:
        print('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E' % therm_args)
    # write data to file
    thermo.write('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n' % therm_args)

def write_traj(traj, natoms, box, x):
    ''' writes trajectory data to traj file '''
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
    
# ---------------------
# monte carlo procedure
# ---------------------

def move_mc(lmps, Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
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
    
def get_sample(x, v, box, el, units, lat, sz, mass, P, dt,
               Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
               dpos, dbox, T, mod):
    ''' performs enough monte carlo moves to generate a sample (determined by mod) '''
    # initialize lammps object
    lmps = lammps_init(x, v, box, el, units, lat, sz, mass, P, dt)
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
    # extract system properties
    natoms, x, v, temp, pe, ke, virial, box, vol = lammps_extract(lmps)
    # close lammps and remove input file
    lmps.close()
    # acceptation ratios
    accpos = np.nan_to_num(np.float64(naccpos)/np.float64(ntrypos))
    accvol = np.nan_to_num(np.float64(naccvol)/np.float64(ntryvol))
    acchmc = np.nan_to_num(np.float64(nacchmc)/np.float64(ntryhmc))
    # update mc params
    dpos, dbox, dt = update_mc_param(dpos, dbox, dt, accpos, accvol, acchmc)
    # return lammps object, tries/acceptation counts, and mc params
    return natoms, x, v, temp, pe, ke, virial, box, vol, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, dt
    
def get_samples(x, v, box, el, units, lat, sz, mass, P, dt,
                Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
                dpos, dbox, T, mod, thermo, traj, verbose):
    ''' performs monte carlo for all configurations to generate new samples '''
    n_press, n_temp = Pf.shape
    # loop through pressures
    for i in xrange(n_press):
        # loop through temperatures
        for j in xrange(n_temp):
            # get new sample configuration for press/temp combo
            dat = get_sample(x[i, j], v[i, j], box[i, j], el, units, lat, sz, mass, P[i], dt[i, j],
                             Et[i, j], Pf[i, j], ppos, pvol, phmc,
                             ntrypos[i, j], naccpos[i, j], ntryvol[i, j], naccvol[i, j], ntryhmc[i, j], nacchmc[i, j],
                             dpos[i, j], dbox[i, j], T[j], mod)
            natoms[i, j], x[i, j], v[i, j] = dat[:3]
            temp[i, j], pe[i, j], ke[i, j], virial[i, j], box[i, j], vol[i, j] = dat[3:9]
            ntrypos[i, j], naccpos[i, j] = dat[9:11]
            ntryvol[i, j], naccvol[i, j] = dat[11:13]
            ntryhmc[i, j], nacchmc[i, j] = dat[13:15]
            dpos[i, j], dbox[i, j], dt[i, j] = dat[15:18]
    # write to data storage files
    if verbose:
        print('\n')
    for i in xrange(n_press):
        for j in xrange(n_temp):
            accpos = np.nan_to_num(np.float64(naccpos[i, j])/np.float64(ntrypos[i, j]))
            accvol = np.nan_to_num(np.float64(naccvol[i, j])/np.float64(ntryvol[i, j]))
            acchmc = np.nan_to_num(np.float64(nacchmc[i, j])/np.float64(ntryhmc[i, j]))
            write_thermo(thermo[i, j], temp[i, j], pe[i, j], ke[i, j], virial[i, j], vol[i, j], accpos, accvol, acchmc, verbose)
            write_traj(traj[i, j], natoms[i, j], box[i, j], x[i, j])
    # return lammps object, tries/acceptation counts, and mc params
    return natoms, x, v, temp, pe, ke, virial, box, vol, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, dt
    
def get_samples_par(client, x, v, box, el, units, lat, sz, mass, P, dt,
                    Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
                    dpos, dbox, T, mod, thermo, traj, verbose):
    ''' performs monte carlo in parallel for all configurations to generate new samples '''
    n_press, n_temp = Pf.shape
    operations = [delayed(get_sample)(x[i, j], v[i, j], box[i, j], el, units, lat, sz, mass, P[i], dt[i, j], 
                                      Et[i, j], Pf[i, j], ppos, pvol, phmc, 
                                      ntrypos[i, j], naccpos[i, j], ntryvol[i, j], naccvol[i, j], ntryhmc[i, j], nacchmc[i, j],
                                      dpos[i, j], dbox[i, j], T[j], mod) for i in xrange(n_press) for j in xrange(n_temp)]
    futures = client.compute(operations)
    if verbose:
        progress(futures)
    k = 0
    for i in xrange(n_press):
        for j in xrange(n_temp):
            dat = futures[k].result()
            k += 1
            natoms[i, j], x[i, j], v[i, j] = dat[:3]
            temp[i, j], pe[i, j], ke[i, j], virial[i, j], box[i, j], vol[i, j] = dat[3:9]
            ntrypos[i, j], naccpos[i, j] = dat[9:11]
            ntryvol[i, j], naccvol[i, j] = dat[11:13]
            ntryhmc[i, j], nacchmc[i, j] = dat[13:15]
            dpos[i, j], dbox[i, j], dt[i, j] = dat[15:18]
    # write to data storage files
    for i in xrange(n_press):
        for j in xrange(n_temp):
            accpos = np.nan_to_num(np.float64(naccpos[i, j])/np.float64(ntrypos[i, j]))
            accvol = np.nan_to_num(np.float64(naccvol[i, j])/np.float64(ntryvol[i, j]))
            acchmc = np.nan_to_num(np.float64(nacchmc[i, j])/np.float64(ntryhmc[i, j]))
            write_thermo(thermo[i, j], temp[i, j], pe[i, j], ke[i, j], virial[i, j], vol[i, j], accpos, accvol, acchmc, verbose)
            write_traj(traj[i, j], natoms[i, j], box[i, j], x[i, j])
    # return lammps object, tries/acceptation counts, and mc params
    return natoms, x, v, temp, pe, ke, virial, box, vol, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, dt
    
# -----------------------------------------
# replica exchange markov chain monte carlo
# -----------------------------------------

def rep_exch(natoms, x, v, temp, pe, ke, virial, box, vol, Et, Pf, verbose):
    ''' performs parallel tempering acrros all samples 
        accepts/rejects based on enthalpy metropolis criterion '''
    # simulation set shape
    n_press, n_temp = Pf.shape
    # flatten lammps objects and constants
    natoms = natoms.reshape(-1)
    x = x.reshape(-1)
    v = v.reshape(-1)
    temp = temp.reshape(-1)
    pe = pe.reshape(-1)
    ke = ke.reshape(-1)
    virial = virial.reshape(-1)
    box = box.reshape(-1)
    vol = vol.reshape(-1)
    Et = Et.reshape(-1)
    Pf = Pf.reshape(-1)
    # catalog swaps and swapping pairs
    swaps = 0
    pairs = []
    # loop through upper right triangular matrix
    for i in xrange(len(Pf)):
        for j in xrange(i+1, len(Pf)):
            # configuration energies
            etoti = pe[i]+ke[i]
            etotj = pe[j]+ke[j]
            # change in enthalpy
            dH = (etoti-etotj)*(1/Et[i]-1/Et[j])+(Pf[i]-Pf[j])*(vol[i]-vol[j])
            if np.random.rand() <= np.min([1, np.exp(dH)]):
                swaps += 1
                pairs.append('(%d, %d)' % (i, j))
                # swap lammps objects
                natoms[j], natoms[i] = natoms[i], natoms[j]
                x[j], x[i] = x[i], x[j]
                v[j], v[i] = v[i], v[j]
                temp[j], temp[i] = temp[i], temp[j]
                pe[j], pe[i] = pe[i], pe[j]
                ke[j], ke[i] = ke[i], ke[j]
                virial[j], virial[i] = virial[i], virial[j]
                box[j], box[i] = box[i], box[j]
                vol[j], vol[i] = vol[i], vol[j]
    if verbose:
        print('%d replica exchanges performed: ' % swaps, ' '.join(pairs))
    natoms = natoms.reshape((n_press, n_temp))
    x = x.reshape((n_press, n_temp))
    v = v.reshape((n_press, n_temp))
    temp = temp.reshape((n_press, n_temp))
    pe = pe.reshape((n_press, n_temp))
    ke = ke.reshape((n_press, n_temp))
    virial = virial.reshape((n_press, n_temp))
    box = box.reshape((n_press, n_temp))
    vol = vol.reshape((n_press, n_temp))
    # return list of lammps objects
    return natoms, x, v, temp, pe, ke, virial, box, vol

# -----------
# build lists
# -----------

# thermo constants
Et = np.zeros((n_press, n_temp), dtype=float)
Pf = np.zeros((n_press, n_temp), dtype=float)
# lammps objects and data storage files
thermo = np.empty((n_press, n_temp), dtype=object)
traj = np.empty((n_press, n_temp), dtype=object)
# system properties
natoms = np.zeros((n_press, n_temp), dtype=int)
x = np.empty((n_press, n_temp), dtype=object)
v = np.empty((n_press, n_temp), dtype=object)
temp = np.zeros((n_press, n_temp), dtype=float)
pe = np.zeros((n_press, n_temp), dtype=float)
ke = np.zeros((n_press, n_temp), dtype=float)
virial = np.zeros((n_press, n_temp), dtype=float)
box = np.zeros((n_press, n_temp), dtype=float)
vol = np.zeros((n_press, n_temp), dtype=float)
# monte carlo tries/acceptations
ntrypos = np.zeros((n_press, n_temp), dtype=float)
naccpos = np.zeros((n_press, n_temp), dtype=float)
ntryvol = np.zeros((n_press, n_temp), dtype=float)
naccvol = np.zeros((n_press, n_temp), dtype=float)
ntryhmc = np.zeros((n_press, n_temp), dtype=float)
nacchmc = np.zeros((n_press, n_temp), dtype=float)
# loop through pressures
for i in xrange(n_press):
    # loop through temperatures
    for j in xrange(n_temp):
        # set thermo constants
        Et[i, j], Pf[i, j] = define_constants(units[el], P[el][i], T[el][j])
        # initialize lammps object and data storage files
        dat = sample_init(i, j, el, units[el], lat[el], sz[el], mass[el], P[el][i], dpos[i, j], dt[i, j])
        natoms[i, j], x[i, j], v[i, j] = dat[:3]
        temp[i, j], pe[i, j], ke[i, j], virial[i, j], box[i, j], vol[i, j] = dat[3:9]
        thermo[i, j], traj[i, j] = dat[9:11]
        # write thermo file header
        thermo_header(thermo[i, j], n_smpl, cutoff, mod, n_swps, ppos, pvol, phmc,
                      n_stps, seed, el, units[el], lat[el], sz[el], mass[el],
                      P[el][i], T[el][j], dt[i, j], dpos[i, j], dbox[i, j])

# -----------                      
# monte carlo
# -----------

if parallel:
    if distributed:
        # construct scheduler with mpi
        sched_init(system, nworkers, path)
        # start client with scheduler file
        client = Client(scheduler=path)
    else:
        # construct local cluster
        cluster = LocalCluster(n_workers=nworkers, threads_per_worker=nthreads, processes=processes)
        # start client with local cluster
        client = Client(cluster)
    # display client information
    if verbose:
        print(client)
# loop through to number of samples that need to be collected
for i in xrange(n_smpl):
    if verbose:
        print('step:', i)
    # collect samples for all configurations
    if parallel:
        dat = get_samples_par(client, x, v, box, el, units[el], lat[el], sz[el], mass[el], P[el], dt,
                              Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
                              dpos, dbox, T[el], mod, thermo, traj, verbose)
        client.restart()  # prevent memory leak
    else:
        dat = get_samples(x, v, box, el, units[el], lat[el], sz[el], mass[el], P[el], dt,
                          Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
                          dpos, dbox, T[el], mod, thermo, traj, verbose)
    # update system data
    natoms, x, v = dat[:3]
    temp, pe, ke, virial, box, vol = dat[3:9]
    ntrypos, naccpos = dat[9:11]
    ntryvol, naccvol = dat[11:13]
    ntryhmc, nacchmc = dat[13:15]
    dpos, dbox, dt = dat[15:18]
    # perform replica exchange markov chain monte carlo (parallel tempering)
    natoms, x, v, temp, pe, ke, virial, box, vol = rep_exch(natoms, x, v, temp, pe, ke, virial, box, vol, Et, Pf, verbose)
if parallel:
    # terminate client after completion
    client.close()

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
    prefix = fpref(name, el, lat[el], P[el][i])
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
            with open(ftraj[i][j], 'r') as fi:
                k = 0
                for line in fi:
                    k += 1
                    if k > (natoms[i, j]+1)*cutoff:
                        fo.write(line)

# --------------
# clean up files
# --------------

# remove all temporary files
for i in xrange(n_press):
    for j in xrange(n_temp):
        os.remove(fthrm[i][j])
        os.remove(ftraj[i][j])   