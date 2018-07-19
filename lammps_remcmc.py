# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 04:20:00 2018

@author: Nicholas
"""

from __future__ import division, print_function
import sys, os, subprocess
import numpy as np
from lammps import lammps
os.environ['DASK_ALLOWED_FAILURES'] = '32'
from distributed import Client, LocalCluster, progress
from dask import delayed

# --------------
# run parameters
# --------------

# boolean for controlling verbosity
if '--verbose' in sys.argv:
    verbose = True
else:
    verbose = False       

# boolean for controlling parallel run
if '--noparallel' in sys.argv:
    parallel = False
else:
    parallel = True
# boolean for choosing distributed or local cluster
if '--distributed' in sys.argv:
    distributed = True
    path = os.getcwd()+'/scheduler.json'  # path for scheduler file
else:
    distributed = False 
# boolean for choosing whether to use processes
if '--noprocesses' in sys.argv:
    processes = False
else:
    processes = True     

# switch for mpirun or aprun
if '--ap' is sys.argv:
    system = 'ap'
else:
    system = 'mpi'
# number of processors
if '--nworker' in sys.argv:
    i = sys.argv.index('--nworker')
    nworker = int(sys.argv[i+1])
else:
    if processes:
        nworker = 16
    else:
        nworker = 1
# threads per worker
if '--nthread' in sys.argv:
    i = sys.argv.index('--nthread')
    nthread = int(sys.argv[i+1])
else:
    if processes:
        nthread = 1
    else:
        nthread = 16

# simulation name
if '--name' in sys.argv:
    i = sys.argv.index('--name')
    name = sys.argv[i+1]
else:
    name = 'remcmc'
# element choice
if '--element' in sys.argv:
    i = sys.argv.index('--element')
    el = sys.argv[i+1]
else:
    el = 'LJ'
if '--size' in sys.argv:
    i = sys.argv.index('--size')
    sz = int(sys.argv[i+1])
else:
    sz = 4

# pressure and temeprature parameters
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
# number of temperature sets
if '--ntemp' in sys.argv:
    i = sys.argv.index('--ntemp')
    ntemp = int(sys.argv[i+1])
else:
    ntemp = 48
# temperature range
if '--rtemp' in sys.argv:
    i = sys.argv.index('--rtemp')
    ltemp = float(sys.argv[i+1])
    htemp = float(sys.argv[i+2])
else:
    ltemp = 0.25
    htemp = 2.5

# monte carlo parameters
# sample cutoff
if '--cutoff' in sys.argv:
    i = sys.argv.index('--cutoff')
    cutoff = int(sys.argv[i+1])
else:
    cutoff = 1024
# number of samples
if '--nsmpl' in sys.argv:
    i = sys.argv.index('--nsmpl')
    nsmpl = cutoff+int(sys.argv[i+1])
else:
    nsmpl = cutoff+1024
# frequency of data storage
if '--mod' in sys.argv:
    i = sys.argv.index('--mod')
    mod = int(sys.argv[i+1])
else:
    mod = 128
nswps = nsmpl*mod  # total mc sweeps
# probability of pos move
if '--ppos' in sys.argv:
    i = sys.argv.index('--ppos')
    ppos = float(sys.argv[i+1])
else:
    ppos = 0.015625
# probability of vol move
if '--pvol' in sys.argv:
    i = sys.argv.index('--pvol')
    pvol = float(sys.argv[i+1])
else:
    pvol = 0.25
phmc = 1-ppos-pvol  # probability of hmc move
# md steps during hmc
if '--nstps' in sys.argv:
    i = sys.argv.index('--nstps')
    nstps = int(sys.argv[i+1])
else:
    nstps = 16

# set random seed
seed = 256
np.random.seed(seed)

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
P = np.linspace(lpress, hpress, npress, dtype=np.float64)
# temperature
T = np.linspace(ltemp, htemp, ntemp, dtype=np.float64)
# lattice type and parameter
lat = {'Ti': ('bcc', 2.951),
       'Al': ('fcc', 4.046),
       'Ni': ('fcc', 3.524),
       'Cu': ('fcc', 3.615),
       'LJ': ('fcc', 1.122)}
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
dbox = 0.03125*lat[el][1]*np.ones((npress, ntemp))
# max pos adjustment
dpos = 0.03125*lat[el][1]*np.ones((npress, ntemp))  
# hmc timestep
dt = timestep[units[el]]*np.ones((npress, ntemp))

# ---------------------
# client initialization
# ---------------------

def schedInit(system, nproc, path):
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

def defineConstants(units, P, T):
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

def lammpsInput(el, units, lat, sz, mass, P, dt):
    ''' constructs input file for lammps
        takes element name, lattice definitions, size, and simulation name
        returns input file name '''
    lj_param = (1.0, 1.0)
    # convert lattice definition list to strings
    prefix = fpref(name, el, lat, P)
    # set lammps file name
    lmpsfilein = prefix+'.in'
    # open lammps file
    lmpsfile = open(lmpsfilein, 'wb')
    # file header
    lmpsfile.write('# LAMMPS Monte Carlo: %s\n\n' % el)
    # units and atom style
    lmpsfile.write('units %s\n' % units)
    lmpsfile.write('atom_style atomic\n')
    lmpsfile.write('atom_modify map yes\n\n')
    # construct simulation box
    lmpsfile.write('boundary p p p\n')
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
    
def lammpsInit(x, v, box, el, units, lat, sz, mass, P, dt):
    ''' initializes system info and data storage files '''
    # generate input file
    lmpsfilein = lammpsInput(el, units, lat, sz, mass, P, dt)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfilein)
    # set system info
    lmps.command('change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box' % (3*(box,)))
    lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
    lmps.scatter_atoms('v', 1, 3, np.ctypeslib.as_ctypes(v))
    lmps.command('run 0')
    return lmps
    
def lammpsExtract(lmps):
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
    
def sampleInit(i, j, el, units, lat, sz, mass, P, dpos, dt):
    ''' initializes system info and data storage files '''
    # generate input file
    lmpsfilein = lammpsInput(el, units, lat, sz, mass, P, dt)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfilein)
    # minimize lattice structure
    lmps.command('unfix 1')
    lmps.command('fix 1 all box/relax iso %f vmax %f' % (P, 0.0009765625))
    lmps.command('minimize 0.0 %f %d %d' % (1.49011612e-8, 1024, 8192))
    lmps.command('displace_atoms all random %f %f %f %d' % (3*(dpos,)+(np.random.randint(1, 2**16),)))
    # extract all system info
    natoms, x, v, temp, pe, ke, virial, box, vol = lammpsExtract(lmps)
    # open data storage files
    thermo = open(lmpsfilein.replace('.in', '%02d%02d.thrm' % (i, j)), 'wb')
    traj = open(lmpsfilein.replace('.in', '%02d%02d.traj' % (i, j)), 'wb')
    lmps.close()
    # return system info and data storage files
    return natoms, x, v, temp, pe, ke, virial, box, vol, thermo, traj
    
# -----------------------------
# output file utility functions
# -----------------------------
    
def thermoHeader(thermo, nsmpl, cutoff, mod, nswps, ppos, pvol, phmc, nstps, 
                 seed, el, units, lat, sz, mass, P, T, dt, dpos, dbox):
    ''' writes header containing simulation information to thermo file '''
    thermo.write('#----------------------\n')
    thermo.write('# simulation parameters\n')
    thermo.write('#----------------------\n')
    thermo.write('# nsmpl:    %d\n' % nsmpl)
    thermo.write('# cutoff:   %d\n' % cutoff)
    thermo.write('# mod:      %d\n' % mod)
    thermo.write('# nswps:    %d\n' % nswps)
    thermo.write('# ppos:     %f\n' % ppos)
    thermo.write('# pvol:     %f\n' % pvol)
    thermo.write('# phmc:     %f\n' % phmc)
    thermo.write('# nstps:    %d\n' % nstps)
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
    thermo.flush()

def writeThermo(thermo, temp, pe, ke, virial, vol, accpos, accvol, acchmc, verbose):
    ''' writes thermodynamic properties to thermo file '''
    # print thermal argument string
    therm_args = (temp, pe, ke, virial, vol, accpos, accvol, acchmc)
    if verbose:
        print('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E' % therm_args)
    # write data to file
    thermo.write('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n' % therm_args)
    thermo.flush()

def writeTraj(traj, natoms, box, x):
    ''' writes trajectory data to traj file '''
    # write data to file
    traj.write('%d %.4E\n' % (natoms, box))
    for k in xrange(natoms):
        traj.write('%.4E %.4E %.4E\n' % tuple(x[3*k:3*k+3]))
    traj.flush()

# -----------------
# monte carlo moves
# -----------------

def positionMC(lmps, Et, ntrypos, naccpos, dpos):
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
    
def volumeMC(lmps, Et, Pf, ntryvol, naccvol, dbox):
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
    dH = (penew-pe)+Pf*(volnew-vol)-(natoms+1)*np.log(volnew/vol)
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
    
def hamiltonianMC(lmps, Et, ntryhmc, nacchmc, T, dt):
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
    lmps.command('run %d' % nstps)  # this part should be implemented as parallel
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

def updateMCParam(dpos, dbox, dt, accpos, accvol, acchmc):
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

def moveMC(lmps, Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
           dpos, dbox, T, dt):
    ''' performs monte carlo moves '''
    roll = np.random.rand()
    # position monte carlo
    if roll <= ppos:
        lmps, ntrypos, naccpos = positionMC(lmps, Et, ntrypos, naccpos, dpos)
    # volume monte carlo
    elif roll <= (ppos+pvol):
        lmps, ntryvol, naccvol = volumeMC(lmps, Et, Pf, ntryvol, naccvol, dbox)
    # hamiltonian monte carlo
    else:
        lmps, ntryhmc, nacchmc = hamiltonianMC(lmps, Et, ntryhmc, nacchmc, T, dt)
    # return lammps object and tries/acceptations counts
    return lmps, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc
    
def getSample(x, v, box, el, units, lat, sz, mass, P, dt,
              Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
              dpos, dbox, T, mod):
    ''' performs enough monte carlo moves to generate a sample (determined by mod) '''
    # initialize lammps object
    lmps = lammpsInit(x, v, box, el, units, lat, sz, mass, P, dt)
    # loop through monte carlo moves
    for i in xrange(mod):
        dat =  moveMC(lmps, Et, Pf,
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
    natoms, x, v, temp, pe, ke, virial, box, vol = lammpsExtract(lmps)
    # close lammps and remove input file
    lmps.close()
    # acceptation ratios
    with np.errstate(invalid='ignore'):
        accpos = np.nan_to_num(np.float64(naccpos)/np.float64(ntrypos))
        accvol = np.nan_to_num(np.float64(naccvol)/np.float64(ntryvol))
        acchmc = np.nan_to_num(np.float64(nacchmc)/np.float64(ntryhmc))
    # update mc params
    dpos, dbox, dt = updateMCParam(dpos, dbox, dt, accpos, accvol, acchmc)
    # return lammps object, tries/acceptation counts, and mc params
    return natoms, x, v, temp, pe, ke, virial, box, vol, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, dt
    
    
def getSamplesPar(client, x, v, box, el, units, lat, sz, mass, P, dt,
                  Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
                  dpos, dbox, T, mod, thermo, traj, verbose):
    ''' performs monte carlo in parallel for all configurations to generate new samples '''
    npress, ntemp = Pf.shape
    operations = [delayed(getSample)(x[i, j], v[i, j], box[i, j], el, units, lat, sz, mass, P[i], dt[i, j], 
                                     Et[i, j], Pf[i, j], ppos, pvol, phmc, 
                                     ntrypos[i, j], naccpos[i, j], ntryvol[i, j], naccvol[i, j], ntryhmc[i, j], nacchmc[i, j],
                                     dpos[i, j], dbox[i, j], T[j], mod) for i in xrange(npress) for j in xrange(ntemp)]
    futures = client.compute(operations)
    if verbose:
        progress(futures)
    statuses = np.array([f.status for f in futures])
    if verbose:
        print('future statuses: ')
        print(statuses)
    if 'error' in statuses:
        if verbose:
            errors = np.array(futures)[statuses != 'finished']
            print('%d calculations unfinished' % errors.size)
        client.recreate_error_locally(futures)
    results = client.gather(futures)
    k = 0
    for i in xrange(npress):
        for j in xrange(ntemp):
            dat = results[k]
            k += 1
            natoms[i, j], x[i, j], v[i, j] = dat[:3]
            temp[i, j], pe[i, j], ke[i, j], virial[i, j], box[i, j], vol[i, j] = dat[3:9]
            ntrypos[i, j], naccpos[i, j] = dat[9:11]
            ntryvol[i, j], naccvol[i, j] = dat[11:13]
            ntryhmc[i, j], nacchmc[i, j] = dat[13:15]
            dpos[i, j], dbox[i, j], dt[i, j] = dat[15:18]
    # write to data storage files
    for i in xrange(npress):
        for j in xrange(ntemp):
            with np.errstate(invalid='ignore'):
                accpos = np.nan_to_num(np.float64(naccpos[i, j])/np.float64(ntrypos[i, j]))
                accvol = np.nan_to_num(np.float64(naccvol[i, j])/np.float64(ntryvol[i, j]))
                acchmc = np.nan_to_num(np.float64(nacchmc[i, j])/np.float64(ntryhmc[i, j]))
            writeThermo(thermo[i, j], temp[i, j], pe[i, j], ke[i, j], virial[i, j], vol[i, j], accpos, accvol, acchmc, verbose)
            writeTraj(traj[i, j], natoms[i, j], box[i, j], x[i, j])
    # return lammps object, tries/acceptation counts, and mc params
    return natoms, x, v, temp, pe, ke, virial, box, vol, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, dt
    
def getSamples(x, v, box, el, units, lat, sz, mass, P, dt,
               Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
               dpos, dbox, T, mod, thermo, traj, verbose):
    ''' performs monte carlo for all configurations to generate new samples '''
    npress, ntemp = Pf.shape
    # loop through pressures
    for i in xrange(npress):
        # loop through temperatures
        for j in xrange(ntemp):
            # get new sample configuration for press/temp combo
            dat = getSample(x[i, j], v[i, j], box[i, j], el, units, lat, sz, mass, P[i], dt[i, j],
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
    for i in xrange(npress):
        for j in xrange(ntemp):
            with np.errstate(invalid='ignore'):
                accpos = np.nan_to_num(np.float64(naccpos[i, j])/np.float64(ntrypos[i, j]))
                accvol = np.nan_to_num(np.float64(naccvol[i, j])/np.float64(ntryvol[i, j]))
                acchmc = np.nan_to_num(np.float64(nacchmc[i, j])/np.float64(ntryhmc[i, j]))
            writeThermo(thermo[i, j], temp[i, j], pe[i, j], ke[i, j], virial[i, j], vol[i, j], accpos, accvol, acchmc, verbose)
            writeTraj(traj[i, j], natoms[i, j], box[i, j], x[i, j])
    # return lammps object, tries/acceptation counts, and mc params
    return natoms, x, v, temp, pe, ke, virial, box, vol, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, dpos, dbox, dt
    
# -----------------------------------------
# replica exchange markov chain monte carlo
# -----------------------------------------

def repExch(natoms, x, v, temp, pe, ke, virial, box, vol, Et, Pf, verbose):
    ''' performs parallel tempering acrros all samples 
        accepts/rejects based on enthalpy metropolis criterion '''
    # simulation set shape
    npress, ntemp = Pf.shape
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
    natoms = natoms.reshape((npress, ntemp))
    x = x.reshape((npress, ntemp))
    v = v.reshape((npress, ntemp))
    temp = temp.reshape((npress, ntemp))
    pe = pe.reshape((npress, ntemp))
    ke = ke.reshape((npress, ntemp))
    virial = virial.reshape((npress, ntemp))
    box = box.reshape((npress, ntemp))
    vol = vol.reshape((npress, ntemp))
    # return list of lammps objects
    return natoms, x, v, temp, pe, ke, virial, box, vol

# -----------
# build lists
# -----------

# thermo constants
Et = np.zeros((npress, ntemp), dtype=float)
Pf = np.zeros((npress, ntemp), dtype=float)
# lammps objects and data storage files
thermo = np.empty((npress, ntemp), dtype=object)
traj = np.empty((npress, ntemp), dtype=object)
# system properties
natoms = np.zeros((npress, ntemp), dtype=int)
x = np.empty((npress, ntemp), dtype=object)
v = np.empty((npress, ntemp), dtype=object)
temp = np.zeros((npress, ntemp), dtype=float)
pe = np.zeros((npress, ntemp), dtype=float)
ke = np.zeros((npress, ntemp), dtype=float)
virial = np.zeros((npress, ntemp), dtype=float)
box = np.zeros((npress, ntemp), dtype=float)
vol = np.zeros((npress, ntemp), dtype=float)
# monte carlo tries/acceptations
ntrypos = np.zeros((npress, ntemp), dtype=float)
naccpos = np.zeros((npress, ntemp), dtype=float)
ntryvol = np.zeros((npress, ntemp), dtype=float)
naccvol = np.zeros((npress, ntemp), dtype=float)
ntryhmc = np.zeros((npress, ntemp), dtype=float)
nacchmc = np.zeros((npress, ntemp), dtype=float)
# loop through pressures
for i in xrange(npress):
    # loop through temperatures
    for j in xrange(ntemp):
        # set thermo constants
        Et[i, j], Pf[i, j] = defineConstants(units[el], P[i], T[j])
        # initialize lammps object and data storage files
        dat = sampleInit(i, j, el, units[el], lat[el], sz, mass[el], P[i], dpos[i, j], dt[i, j])
        natoms[i, j], x[i, j], v[i, j] = dat[:3]
        temp[i, j], pe[i, j], ke[i, j], virial[i, j], box[i, j], vol[i, j] = dat[3:9]
        thermo[i, j], traj[i, j] = dat[9:11]
        # write thermo file header
        thermoHeader(thermo[i, j], nsmpl, cutoff, mod, nswps, ppos, pvol, phmc,
                      nstps, seed, el, units[el], lat[el], sz, mass[el],
                      P[i], T[j], dt[i, j], dpos[i, j], dbox[i, j])

# -----------                      
# monte carlo
# -----------

if parallel:
    if distributed:
        # construct scheduler with mpi
        schedInit(system, nworker, path)
        # start client with scheduler file
        client = Client(scheduler=path)
    else:
        # construct local cluster
        cluster = LocalCluster(n_workers=nworker, threads_per_worker=nthread, processes=processes)
        # start client with local cluster
        client = Client(cluster)
    # display client information
    if verbose:
        print(client.scheduler_info)
# loop through to number of samples that need to be collected
for i in xrange(nsmpl):
    if verbose:
        print('step:', i)
    # collect samples for all configurations
    if parallel:
        dat = getSamplesPar(client, x, v, box, el, units[el], lat[el], sz, mass[el], P, dt,
                            Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
                            dpos, dbox, T, mod, thermo, traj, verbose)
    else:
        dat = getSamples(x, v, box, el, units[el], lat[el], sz, mass[el], P, dt,
                         Et, Pf, ppos, pvol, phmc, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
                         dpos, dbox, T, mod, thermo, traj, verbose)
    # update system data
    natoms, x, v = dat[:3]
    temp, pe, ke, virial, box, vol = dat[3:9]
    ntrypos, naccpos = dat[9:11]
    ntryvol, naccvol = dat[11:13]
    ntryhmc, nacchmc = dat[13:15]
    dpos, dbox, dt = dat[15:18]
    # perform replica exchange markov chain monte carlo (parallel tempering)
    natoms, x, v, temp, pe, ke, virial, box, vol = repExch(natoms, x, v, temp, pe, ke, virial, box, vol, Et, Pf, verbose)
if parallel:
    # terminate client after completion
    client.close()

# ------------------
# final data storage
# ------------------

# loop through pressures
for i in xrange(npress):
    # loop through temperatures
    for j in xrange(ntemp):
        thermo[i, j].close()
        traj[i, j].close()

# construct data storage file name lists
fthrm = [[thermo[i, j].name for j in xrange(ntemp)] for i in xrange(npress)]
ftraj = [[traj[i, j].name for j in xrange(ntemp)] for i in xrange(npress)]
# loop through pressures
for i in xrange(npress):
    # get prefix
    prefix = fpref(name, el, lat[el], P[i])
    # open collected thermo data file
    with open(prefix+'.thrm', 'wb') as fo:
        # write data to collected thermo file
        for j in xrange(ntemp):
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
    with open(prefix+'.traj', 'wb') as fo:
        # write data to collected traj file
        for j in xrange(ntemp):
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
for i in xrange(npress):
    for j in xrange(ntemp):
        os.remove(fthrm[i][j])
        os.remove(ftraj[i][j])   