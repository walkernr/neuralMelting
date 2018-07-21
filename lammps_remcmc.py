# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 04:20:00 2018

@author: Nicholas
"""

from __future__ import division, print_function
import argparse, os
import numpy as np
import numba as nb
from lammps import lammps

# --------------
# run parameters
# --------------

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
parser.add_argument('-p', '--parallel', help='parallel run', action='store_true')
parser.add_argument('-d', '--distributed', help='distributed run', action='store_true')
parser.add_argument('-q', '--queue', help='submission queue', type=str, default='lasigma')
parser.add_argument('-a', '--allocation', help='submission allocation', type=str, default='hpc_lasigma01')
parser.add_argument('-nn', '--nodes', help='number of nodes', type=int, default=1)
parser.add_argument('-np', '--procs_per_node', help='number of processors per node', type=int, default=16)
parser.add_argument('-w', '--walltime', help='job walltime', type=int, default=72)
parser.add_argument('-m', '--memory', help='total job memory', type=int, default=32)
parser.add_argument('-nw', '--workers', help='total job worker count', type=int, default=4)
parser.add_argument('-nt', '--threads', help='threads per worker', type=int, default=1)
parser.add_argument('-n', '--name', help='name of simulation', type=str, default='test')
parser.add_argument('-e', '--element', help='element choice', type=str, default='LJ')
parser.add_argument('-ss', '--supercell_size', help='supercell size', type=int, default=4)
parser.add_argument('-pn', '--pressure_number', help='number of pressures', type=int, default=4)
parser.add_argument('-pr', '--pressure_range', help='pressure range', type=float, nargs=2, default=[2, 8])
parser.add_argument('-tn', '--temperature_number', help='number of temperatures', type=int, default=32)
parser.add_argument('-tr', '--temperature_range', help='temperature range', type=float, nargs=2, default=[0.25, 2.5])
parser.add_argument('-sc', '--sample_cutoff', help='sample cutoff', type=int, default=0)
parser.add_argument('-sn', '--sample_number', help='sample number', type=int, default=4)
parser.add_argument('-sm', '--sample_mod', help='sample record modulo', type=int, default=32)
parser.add_argument('-pm', '--position_move', help='position monte carlo move probability', type=float, default=0.015625)
parser.add_argument('-vm', '--volume_move', help='volume monte carlo move probability', type=float, default=0.25)
parser.add_argument('-t', '--timesteps', help='hamiltonian monte carlo timesteps', type=int, default=8)

args = parser.parse_args()

verbose = args.verbose
parallel = args.parallel
distributed = args.distributed
queue = args.queue
alloc = args.allocation
nodes = args.nodes
ppn = args.procs_per_node
walltime = args.walltime
mem = args.memory
nworker = args.workers
nthread = args.threads
name = args.name
el = args.element
sz = args.supercell_size
npress = args.pressure_number
lpress, hpress = args.pressure_range
ntemp = args.temperature_number
ltemp, htemp = args.temperature_range
cutoff = args.sample_cutoff
nsmpl = args.sample_number
mod = args.sample_mod
ppos = args.position_move
pvol = args.volume_move
nstps = args.timesteps

if parallel:
    os.environ['DASK_ALLOWED_FAILURES'] = '4'
    from distributed import Client, LocalCluster, progress
    from dask import delayed
if distributed:
    import time
    from dask_jobqueue import PBSCluster

nsmpl = cutoff+nsmpl
nswps = nsmpl*mod
phmc = 1-ppos-pvol

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
    return os.getcwd()+'/'+prefix

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
    with open(lmpsfilein, 'wb') as fo:
        # file header
        fo.write('# LAMMPS Monte Carlo: %s\n\n' % el)
        # units and atom style
        fo.write('units %s\n' % units)
        fo.write('atom_style atomic\n')
        fo.write('atom_modify map yes\n\n')
        # construct simulation box
        fo.write('boundary p p p\n')
        fo.write('lattice %s %s\n' % tuple(lat))
        fo.write('region box block 0 %d 0 %d 0 %d\n' % (3*(sz,)))
        fo.write('create_box 1 box\n')
        fo.write('create_atoms 1 box\n\n')
        # potential definitions
        if el == 'Ti':
            fo.write('pair_style meam/c\n')
            fo.write('mass 1 47.867\n')
            fo.write('pair_coeff * * library.meam Ti Al TiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % el)
        if el == 'Al':
            fo.write('pair_style meam/c\n')
            fo.write('mass 1 %f\n' % mass)
            fo.write('pair_coeff * * library.meam Ti Al TiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % el)
        if el == 'Ni':
            fo.write('pair_style meam/c\n')
            fo.write('mass 1 %f\n' % mass)
            fo.write('pair_coeff * * library.Ni.meam Ni Ni.meam %s\n\n' % el)
        if el == 'Cu':
            fo.write('pair_style meam/c\n')
            fo.write('mass 1 %f\n' % mass)
            fo.write('pair_coeff * * library.Cu.meam Cu Cu.meam %s\n\n' % el)
        if el == 'LJ':
            fo.write('pair_style lj/cut 2.5\n')
            fo.write('mass 1 %f\n' % mass)
            fo.write('pair_coeff 1 1 %f %f 2.5\n\n' % lj_param)
        # compute kinetic energy
        fo.write('compute thermo_ke all ke\n\n')
        # initialize
        fo.write('timestep %f\n' % dt)
        fo.write('fix 1 all nve\n')
        fo.write('run 0')
    # return name
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
    # thermo = open(lmpsfilein.replace('.in', '%02d%02d.thrm' % (i, j)), 'wb')
    # traj = open(lmpsfilein.replace('.in', '%02d%02d.traj' % (i, j)), 'wb')
    thermo = lmpsfilein.replace('.in', '%02d%02d.thrm' % (i, j))
    traj = lmpsfilein.replace('.in', '%02d%02d.traj' % (i, j))
    lmps.close()
    # return system info and data storage files
    return natoms, x, v, temp, pe, ke, virial, box, vol, thermo, traj
    
# -----------------------------
# output file utility functions
# -----------------------------
    
def thermoHeader(thermo, nsmpl, cutoff, mod, nswps, ppos, pvol, phmc, nstps, 
                 seed, el, units, lat, sz, mass, P, T, dt, dpos, dbox):
    ''' writes header containing simulation information to thermo file '''
    with open(thermo, 'wb') as fo:
        fo.write('#----------------------\n')
        fo.write('# simulation parameters\n')
        fo.write('#----------------------\n')
        fo.write('# nsmpl:    %d\n' % nsmpl)
        fo.write('# cutoff:   %d\n' % cutoff)
        fo.write('# mod:      %d\n' % mod)
        fo.write('# nswps:    %d\n' % nswps)
        fo.write('# ppos:     %f\n' % ppos)
        fo.write('# pvol:     %f\n' % pvol)
        fo.write('# phmc:     %f\n' % phmc)
        fo.write('# nstps:    %d\n' % nstps)
        fo.write('# seed:     %d\n' % seed)
        fo.write('#----------------------\n')
        fo.write('# material properties\n')
        fo.write('#----------------------\n')
        fo.write('# element:  %s\n' % el)
        fo.write('# units:    %s\n' % units)
        fo.write('# lattice:  %s\n' % lat[0])
        fo.write('# latpar:   %f\n' % lat[1])
        fo.write('# size:     %d\n' % sz)
        fo.write('# mass:     %f\n' % mass)
        fo.write('# press:    %f\n' % P)
        fo.write('# temp:     %f\n' % T)
        fo.write('# dposmax:  %f\n' % dpos)
        fo.write('# dboxmax:  %f\n' % dbox)
        fo.write('# timestep: %f\n' % dt)
        fo.write('# ------------------------------------------------------------\n')
        fo.write('# | temp | pe | ke | virial | vol | accpos | accvol | acchmc |\n')
        fo.write('# ------------------------------------------------------------\n')

def writeThermo(thermo, temp, pe, ke, virial, vol, accpos, accvol, acchmc):
    ''' writes thermodynamic properties to thermo file '''
    # write data to file
    therm_args = (temp, pe, ke, virial, vol, accpos, accvol, acchmc)
    with open(thermo, 'ab') as fo:
        fo.write('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n' % therm_args)

def writeTraj(traj, natoms, box, x):
    ''' writes trajectory data to traj file '''
    # write data to file
    with open(traj, 'ab') as fo:
        fo.write('%d %.4E\n' % (natoms, box))
        for k in xrange(natoms):
            fo.write('%.4E %.4E %.4E\n' % tuple(x[3*k:3*k+3]))

# -----------------
# monte carlo moves
# -----------------

def bulkPositionMC(lmps, Et, ntrypos, naccpos, dpos):
    ''' classic position monte carlo 
        simultaneous random displacement of atoms
        accepts/rejects based on energy metropolis criterion '''
    ntrypos += 1
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    pe = lmps.extract_compute('thermo_pe', None, 0)/Et
    lmps.command('displace_atoms all random %f %f %f %d' % (3*(dpos,)+(np.random.randint(1, 2**16),)))
    lmps.command('run 0')
    xnew = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
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

def iterPositionMC(lmps, Et, ntrypos, naccpos, dpos):
    ''' classic position monte carlo 
        iterative random displacement of atoms
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
        lmps, ntrypos, naccpos = bulkPositionMC(lmps, Et, ntrypos, naccpos, dpos)
        # lmps, ntrypos, naccpos = iterPositionMC(lmps, Et, ntrypos, naccpos, dpos)
    # volume monte carlo
    elif roll <= (ppos+pvol):
        lmps, ntryvol, naccvol = volumeMC(lmps, Et, Pf, ntryvol, naccvol, dbox)
    # hamiltonian monte carlo
    else:
        lmps, ntryhmc, nacchmc = hamiltonianMC(lmps, Et, ntryhmc, nacchmc, T, dt)
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
    return natoms, x, v, temp, pe, ke, virial, box, vol, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, accpos, accvol, acchmc, dpos, dbox, dt
    
    
def getSamplesPar(client, x, v, box, el, units, lat, sz, mass, P, dt,
                  Et, Pf, ppos, pvol, phmc,
                  ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, accpos, accvol, acchmc,
                  dpos, dbox, T, mod, thermo, traj, verbose):
    ''' performs monte carlo in parallel for all configurations to generate new samples '''
    npress, ntemp = Pf.shape
    # list of delayed operations
    operations = [delayed(getSample)(x[i, j], v[i, j], box[i, j], el, units, lat, sz, mass, P[i], dt[i, j], 
                                     Et[i, j], Pf[i, j], ppos, pvol, phmc, 
                                     ntrypos[i, j], naccpos[i, j], ntryvol[i, j], naccvol[i, j], ntryhmc[i, j], nacchmc[i, j],
                                     dpos[i, j], dbox[i, j], T[j], mod) for i in xrange(npress) for j in xrange(ntemp)]
    # submit futures to client
    futures = client.compute(operations)
    # progress bar
    if verbose:
        print('performing monte carlo')
        progress(futures)
    # gather results from workers
    results = client.gather(futures, errors='raise')
    # update system properties and monte carlo parameters
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
            accpos[i, j], accvol[i, j], acchmc[i, j] = dat[15:18]
            dpos[i, j], dbox[i, j], dt[i, j] = dat[18:21]
    # write to data storage files
    operations = [delayed(writeThermo)(thermo[i, j], temp[i, j], pe[i, j], ke[i, j], virial[i, j], vol[i, j],
                                       accpos[i, j], accvol[i, j], acchmc[i, j]) for i in xrange(npress) for j in xrange(ntemp)]
    futures = client.compute(operations)
    if verbose:
        print('\nwriting thermo data')
        progress(futures)
    operations = [delayed(writeTraj)(traj[i, j], natoms[i, j], box[i, j], x[i, j]) for i in xrange(npress) for j in xrange(ntemp)]
    futures = client.compute(operations)
    if verbose:
        print('\nwriting traj data')
        progress(futures)
    # if verbose:
        # print('\n')
        # for i in xrange(npress):
            # for j in xrange(ntemp):
                # therm_args = (temp[i, j], pe[i, j], ke[i, j], virial[i, j], vol[i, j], accpos[i, j], accvol[i, j], acchmc[i, j])
                # print('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E' % therm_args)
    # return lammps object, tries/acceptation counts, and mc params
    return client, natoms, x, v, temp, pe, ke, virial, box, vol, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, accpos, accvol, acchmc, dpos, dbox, dt
    
def getSamples(x, v, box, el, units, lat, sz, mass, P, dt,
               Et, Pf, ppos, pvol, phmc,
               ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, accpos, accvol, acchmc,
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
            accpos[i, j], accvol[i, j], acchmc[i, j] = dat[15:18]
            dpos[i, j], dbox[i, j], dt[i, j] = dat[18:21]
            dpos[i, j], dbox[i, j], dt[i, j] = dat[15:18]
    # write to data storage files
    for i in xrange(npress):
        for j in xrange(ntemp):
            # if verbose:
                # therm_args = (temp[i, j], pe[i, j], ke[i, j], virial[i, j], vol[i, j], accpos[i, j], accvol[i, j], acchmc[i, j])
                # print('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E' % therm_args)
            writeThermo(thermo[i, j], temp[i, j], pe[i, j], ke[i, j], virial[i, j], vol[i, j], accpos[i, j], accvol[i, j], acchmc[i, j])
            writeTraj(traj[i, j], natoms[i, j], box[i, j], x[i, j])
    # return lammps object, tries/acceptation counts, and mc params
    return natoms, x, v, temp, pe, ke, virial, box, vol, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, accpos, accvol, acchmc, dpos, dbox, dt
    
# -----------------------------------------
# replica exchange markov chain monte carlo
# -----------------------------------------

def repExch(natoms, x, v, temp, pe, ke, virial, box, vol, Et, Pf, verbose):
    ''' performs parallel tempering acrros all samples 
        accepts/rejects based on enthalpy metropolis criterion '''
    # simulation set shape
    npress, ntemp = Pf.shape
    # flatten system properties and constants
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
        print('\n%d replica exchanges performed' % swaps)
    # reshape system properties and constants
    natoms = natoms.reshape((npress, ntemp))
    x = x.reshape((npress, ntemp))
    v = v.reshape((npress, ntemp))
    temp = temp.reshape((npress, ntemp))
    pe = pe.reshape((npress, ntemp))
    ke = ke.reshape((npress, ntemp))
    virial = virial.reshape((npress, ntemp))
    box = box.reshape((npress, ntemp))
    vol = vol.reshape((npress, ntemp))
    # return system properties and constants
    return natoms, x, v, temp, pe, ke, virial, box, vol

# -------------
# allocate data
# -------------

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
accpos = np.zeros((npress, ntemp), dtype=float)
accvol = np.zeros((npress, ntemp), dtype=float)
acchmc = np.zeros((npress, ntemp), dtype=float)

#------------------
# initialize client
#------------------

if parallel:
    if distributed:
        # construct distributed cluster
        cluster = PBSCluster(queue=queue, project=alloc, resource_spec='nodes=%d:ppn=%d' % (nodes, ppn), walltime='%d:00:00' % walltime,
                             processes=nworker, cores=nthread*nworker, memory=str(mem)+'GB')
        cluster.start_workers(1)
        # start client with distributed cluster
        client = Client(cluster)
        while 'processes=0 cores=0' in str(client.scheduler_info):
            time.sleep(5)
            if verbose:
                print(client.scheduler_info)
    else:
        # construct local cluster
        cluster = LocalCluster(n_workers=nworker, threads_per_worker=nthread)
        # start client with local cluster
        client = Client(cluster)
        # display client information
        if verbose:
            print(client.scheduler_info)

#-------------------
# initialize samples
#-------------------

if parallel:
    operations = [delayed(defineConstants)(units[el], P[i], T[j]) for i in xrange(npress) for j in xrange(ntemp)]
    futures = client.compute(operations)
    if verbose:
        print('setting constants')
        progress(futures)
    results = client.gather(futures)
    k = 0
    for i in xrange(npress):
        for j in xrange(ntemp):
            Et[i, j], Pf[i, j] = results[k]
            k += 1
    operations = [delayed(sampleInit)(i, j, el, units[el], lat[el], sz, mass[el],
                                      P[i], dpos[i, j], dt[i, j]) for i in xrange(npress) for j in xrange(ntemp)]
    futures = client.compute(operations)
    if verbose:
        print('\ninitializing samples')
        progress(futures)
    results = client.gather(futures)
    k = 0
    for i in xrange(npress):
        for j in xrange(ntemp):
            dat = results[k]
            k += 1
            natoms[i, j], x[i, j], v[i, j] = dat[:3]
            temp[i, j], pe[i, j], ke[i, j], virial[i, j], box[i, j], vol[i, j] = dat[3:9]
            thermo[i, j], traj[i, j] = dat[9:11]
    operations = [delayed(thermoHeader)(thermo[i, j], nsmpl, cutoff, mod, nswps, ppos, pvol, phmc,
                                        nstps, seed, el, units[el], lat[el], sz, mass[el],
                                        P[i], T[j], dt[i, j], dpos[i, j], dbox[i, j]) for i in xrange(npress) for j in xrange(ntemp)]
    futures = client.compute(operations)
    if verbose:
        print('\nwriting thermo headers')
        progress(futures)
    del futures
    del results
else:
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

# loop through to number of samples that need to be collected
for i in xrange(nsmpl):
    if verbose:
        print('step:', i)
    # collect samples for all configurations
    if parallel:
        dat = getSamplesPar(client, x, v, box, el, units[el], lat[el], sz, mass[el], P, dt,
                            Et, Pf, ppos, pvol, phmc,
                            ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, accpos, accvol, acchmc,
                            dpos, dbox, T, mod, thermo, traj, verbose)
        client = dat[0]
        dat = dat[1:]
    else:
        dat = getSamples(x, v, box, el, units[el], lat[el], sz, mass[el], P, dt,
                         Et, Pf, ppos, pvol, phmc,
                         ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc, accpos, accvol, acchmc,
                         dpos, dbox, T, mod, thermo, traj, verbose)
    # update system data
    natoms, x, v = dat[:3]
    temp, pe, ke, virial, box, vol = dat[3:9]
    ntrypos, naccpos = dat[9:11]
    ntryvol, naccvol = dat[11:13]
    ntryhmc, nacchmc = dat[13:15]
    accpos, accvol, acchmc = dat[15:18]
    dpos, dbox, dt = dat[18:21]
    # perform replica exchange markov chain monte carlo (parallel tempering)
    natoms, x, v, temp, pe, ke, virial, box, vol = repExch(natoms, x, v, temp, pe, ke, virial, box, vol, Et, Pf, verbose)
if parallel:
    # terminate client after completion
    client.close()

# ------------------
# final data storage
# ------------------

if verbose:
    print('consolidating files')
# loop through pressures
for i in xrange(npress):
    # get prefix
    prefix = fpref(name, el, lat[el], P[i])
    # open collected thermo data file
    with open(prefix+'.thrm', 'wb') as fo:
        # write data to collected thermo file
        for j in xrange(ntemp):
            with open(thermo[i, j], 'r') as fi:
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
            with open(traj[i, j], 'r') as fi:
                k = 0
                for line in fi:
                    k += 1
                    if k > (natoms[i, j]+1)*cutoff:
                        fo.write(line)

# --------------
# clean up files
# --------------

if verbose:
    print('cleaning files')
# remove all temporary files
for i in xrange(npress):
    for j in xrange(ntemp):
        os.remove(thermo[i, j])
        os.remove(traj[i, j])   