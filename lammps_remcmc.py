# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 04:20:00 2018

@author: Nicholas
"""

from __future__ import division, print_function
import argparse
import os
import numpy as np
from lammps import lammps

# --------------
# run parameters
# --------------

# parse command line (help option generated automatically)

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
parser.add_argument('-p', '--parallel', help='parallel run', action='store_true')
parser.add_argument('-d', '--distributed', help='distributed run', action='store_true')
parser.add_argument('-q', '--queue', help='submission queue', type=str, default='lasigma')
parser.add_argument('-a', '--allocation', help='submission allocation', type=str, default='hpc_lasigma01')
parser.add_argument('-nn', '--nodes', help='number of nodes', type=int, default=1)
parser.add_argument('-np', '--procs_per_node', help='number of processors per node', type=int, default=16)
parser.add_argument('-w', '--walltime', help='job walltime', type=int, default=24)
parser.add_argument('-m', '--memory', help='total job memory', type=int, default=32)
parser.add_argument('-nw', '--workers', help='total job worker count', type=int, default=4)
parser.add_argument('-nt', '--threads', help='threads per worker', type=int, default=1)
parser.add_argument('-n', '--name', help='name of simulation', type=str, default='test')
parser.add_argument('-e', '--element', help='element choice', type=str, default='LJ')
parser.add_argument('-ss', '--supercell_size', help='supercell size', type=int, default=4)
parser.add_argument('-pn', '--pressure_number', help='number of pressures', type=int, default=4)
parser.add_argument('-pr', '--pressure_range', help='pressure range', type=float, nargs=2, default=[2, 8])
parser.add_argument('-tn', '--temperature_number', help='number of temperatures', type=int, default=4)
parser.add_argument('-tr', '--temperature_range', help='temperature range', type=float, nargs=2, default=[0.25, 2.5])
parser.add_argument('-sc', '--sample_cutoff', help='sample cutoff', type=int, default=0)
parser.add_argument('-sn', '--sample_number', help='sample number', type=int, default=4)
parser.add_argument('-sm', '--sample_mod', help='sample record modulo', type=int, default=32)
parser.add_argument('-pm', '--position_move', help='position monte carlo move probability', type=float, default=0.015625)
parser.add_argument('-vm', '--volume_move', help='volume monte carlo move probability', type=float, default=0.25)
parser.add_argument('-t', '--timesteps', help='hamiltonian monte carlo timesteps', type=int, default=8)

args = parser.parse_args()

# booleans for run type
VERBOSE = args.verbose
PARALLEL = args.parallel
DISTRIBUTED = args.distributed
# arguments for distributed run using pbs
QUEUE = args.queue
ALLOC = args.allocation
NODES = args.nodes
PPN = args.procs_per_node
WALLTIME = args.walltime
MEM = args.memory
# arguments for parallel run
NWORKER = args.workers
NTHREAD = args.threads
# simulation name
NAME = args.name
# element choice
EL = args.element
# supercell size in lattice parameters
SZ = args.supercell_size
# pressure parameters
NPRESS = args.pressure_number
LPRESS, HPRESS = args.pressure_range
# temperature parameters
NTEMP = args.temperature_number
LTEMP, HTEMP = args.temperature_range
# sample cutoff
CUTOFF = args.sample_cutoff
# number of recording samples
NSMPL = args.sample_number
# record frequency
MOD = args.sample_mod
# monte carlo probabilities
PPOS = args.position_move
PVOL = args.volume_move
# number of hamiltonian monte carlo timesteps
NSTPS = args.timesteps

if PARALLEL:
    os.environ['DASK_ALLOWED_FAILURES'] = '4'
    from distributed import Client, LocalCluster, progress
    from dask import delayed
if DISTRIBUTED:
    import time
    from dask_jobqueue import PBSCluster

# adjust total samples by cutoff
NSMPL = CUTOFF+NSMPL
# total number of monte carlo sweeps
NSWPS = NSMPL*MOD
# hamiltonian monte carlo probability
PHMC = 1-PPOS-PVOL

# set random seed
SEED = 256
np.random.seed(SEED)

# -------------------
# material properties
# -------------------

# unit system
UNITS = {'Ti': 'metal',
         'Al': 'metal',
         'Ni': 'metal',
         'Cu': 'metal',
         'LJ': 'lj'}
# pressure
P = np.linspace(LPRESS, HPRESS, NPRESS, dtype=np.float64)
# temperature
T = np.linspace(LTEMP, HTEMP, NTEMP, dtype=np.float64)
# lattice type and parameter
LAT = {'Ti': ('bcc', 2.951),
       'Al': ('fcc', 4.046),
       'Ni': ('fcc', 3.524),
       'Cu': ('fcc', 3.615),
       'LJ': ('fcc', 1.122)}
# mass
MASS = {'Ti': 47.867,
        'Al': 29.982,
        'Ni': 58.693,
        'Cu': 63.546,
        'LJ': 1.0}
# timestep
TIMESTEP = {'real': 4.0,
            'metal': 0.00390625,
            'lj': 0.00390625}

# ----------------
# unit definitions
# ----------------


def defineConstants(i, j):
    ''' sets thermodynamic constants according to chosen unit system '''
    if UNITS[EL] == 'real':
        NA = 6.0221409e23                              # avagadro number [num/mol]
        KB = 3.29983e-27                               # boltzmann constant [kcal/K]
        R = kB*NA                                      # gas constant [kcal/(mol K)]
        ET = R*T[j]                                    # thermal energy [kcal/mol]
        PF = 1e-30*(1.01325e5*P[i])/(4.184e3*KB*T[j])  # metropolis prefactor [1/A^3]
    if UNITS[EL] == 'metal':
        KB = 8.61733e-5                                # boltzmann constant [eV/K]
        ET = KB*T[i]                                   # thermal energy [eV]
        PF = 1e-30*(1e5*P[i])/(1.60218e-19*KB*T[j])    # metropolis prefactor [1/A^3]
    if UNITS[EL] == 'lj':
        KB = 1.0                                       # boltzmann constant (normalized and unitless)
        ET = KB*T[j]                                   # thermal energy [T*]
        PF = P[i]/(KB*T[j])                            # metropolis prefactor [1/r*^3]
    return ET, PF

# ---------------------------------
# lammps file/object initialization
# ---------------------------------


def fpref(i):
    ''' returns file prefix for simulation '''
    prefix = '%s.%s.%s.%d.lammps' % (NAME, EL.lower(), LAT[EL][0], int(P[i]))
    return os.getcwd()+'/'+prefix


def lammpsInput(i, j):
    ''' constructs input file for lammps
        takes element name, lattice definitions, size, and simulation name
        returns input file name '''
    lj_param = (1.0, 1.0)
    # convert lattice definition list to strings
    prefix = fpref(i)
    # set lammps file name
    lmpsfilein = prefix+'.in'
    # open lammps file
    with open(lmpsfilein, 'wb') as fo:
        # file header
        fo.write('# LAMMPS Monte Carlo: %s\n\n' % EL)
        # units and atom style
        fo.write('units %s\n' % UNITS[EL])
        fo.write('atom_style atomic\n')
        fo.write('atom_modify map yes\n\n')
        # construct simulation box
        fo.write('boundary p p p\n')
        fo.write('lattice %s %s\n' % tuple(LAT[EL]))
        fo.write('region box block 0 %d 0 %d 0 %d\n' % (3*(SZ,)))
        fo.write('create_box 1 box\n')
        fo.write('create_atoms 1 box\n\n')
        # potential definitions
        if EL == 'Ti':
            fo.write('pair_style meam/c\n')
            fo.write('mass 1 47.867\n')
            fo.write('pair_coeff * * library.meam Ti Al TiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % EL)
        if EL == 'Al':
            fo.write('pair_style meam/c\n')
            fo.write('mass 1 %f\n' % MASS[EL])
            fo.write('pair_coeff * * library.meam Ti Al TiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % EL)
        if EL == 'Ni':
            fo.write('pair_style meam/c\n')
            fo.write('mass 1 %f\n' % MASS[EL])
            fo.write('pair_coeff * * library.Ni.meam Ni Ni.meam %s\n\n' % EL)
        if EL == 'Cu':
            fo.write('pair_style meam/c\n')
            fo.write('mass 1 %f\n' % MASS[EL])
            fo.write('pair_coeff * * library.Cu.meam Cu Cu.meam %s\n\n' % EL)
        if EL == 'LJ':
            fo.write('pair_style lj/cut 2.5\n')
            fo.write('mass 1 %f\n' % MASS[EL])
            fo.write('pair_coeff 1 1 %f %f 2.5\n\n' % lj_param)
        # compute kinetic energy
        fo.write('compute thermo_ke all ke\n\n')
        # initialize
        fo.write('timestep %f\n' % dt[i, j])
        fo.write('fix 1 all nve\n')
        fo.write('run 0')
    # return file name
    return lmpsfilein


def lammpsExtract(lmps):
    ''' extract system properties from lammps object '''
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
    res = natoms, x, v, temp, pe, ke, virial, box, vol
    return res


def sampleInit(i, j):
    ''' initializes system info and data storage files '''
    # generate input file
    lmpsfilein = lammpsInput(i, j)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfilein)
    # minimize lattice structure
    lmps.command('unfix 1')
    lmps.command('fix 1 all box/relax iso %f vmax %f' % (P[i], 0.0009765625))
    lmps.command('minimize 0.0 %f %d %d' % (1.49011612e-8, 1024, 8192))
    lmps.command('displace_atoms all random %f %f %f %d' % (3*(dpos[i, j],)+(np.random.randint(1, 2**16),)))
    # extract all system info
    natoms, x, v, temp, pe, ke, virial, box, vol = lammpsExtract(lmps)
    # open data storage files
    # thermo = open(lmpsfilein.replace('.in', '%02d%02d.thrm' % (i, j)), 'wb')
    # traj = open(lmpsfilein.replace('.in', '%02d%02d.traj' % (i, j)), 'wb')
    thermo = lmpsfilein.replace('.in', '%02d%02d.thrm' % (i, j))
    traj = lmpsfilein.replace('.in', '%02d%02d.traj' % (i, j))
    try:
        os.remove(thermo)
    except:
        pass
    try:
        os.remove(traj)
    except:
        pass
    lmps.close()
    # return system info and data storage files
    res = natoms, x, v, temp, pe, ke, virial, box, vol, thermo, traj
    return res


def lammpsInit(x, v, box, name, el, units, lat, sz, mass, P, dt):
    ''' initializes system info and data storage files '''
    # generate input file
    lmpsfilein = lammpsInput(name, el, units, lat, sz, mass, P, dt)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfilein)
    # set system info
    lmps.command('change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box' % (3*(box,)))
    lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
    lmps.scatter_atoms('v', 1, 3, np.ctypeslib.as_ctypes(v))
    lmps.command('run 0')
    return lmps

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
    res = lmps, ntrypos, naccpos
    return res


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
    res = lmps, ntrypos, naccpos
    return res


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
    res = lmps, ntryvol, naccvol
    return res


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
    res = lmps, ntryhmc, nacchmc
    return res

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
    res = dpos, dbox, dt
    return res

# ---------------------
# monte carlo procedure
# ---------------------


def moveMC(lmps, Et, Pf, ppos, pvol, phmc,
           ntrypos, naccpos,
           ntryvol, naccvol,
           ntryhmc, nacchmc,
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
    res = lmps, ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc
    return res


def getSample(x, v, box, name, el, units, lat, sz, mass, P, dt,
              Et, Pf, ppos, pvol, phmc,
              ntrypos, naccpos,
              ntryvol, naccvol,
              ntryhmc, nacchmc,
              dpos, dbox, T, mod):
    ''' performs enough monte carlo moves to generate a sample (determined by mod) '''
    # initialize lammps object
    lmps = lammpsInit(x, v, box, name, el, units, lat, sz, mass, P, dt)
    # loop through monte carlo moves
    for i in xrange(mod):
        dat = moveMC(lmps, Et, Pf,
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
    res = (natoms, x, v, temp, pe, ke, virial, box, vol,
           ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
           accpos, accvol, acchmc, dpos, dbox, dt)
    return res


def getSamplesPar(x, v, box, name, el, units, lat, sz, mass, P, dt,
                  Et, Pf, ppos, pvol, phmc,
                  ntrypos, naccpos,
                  ntryvol, naccvol,
                  ntryhmc, nacchmc,
                  accpos, accvol, acchmc,
                  dpos, dbox, T, mod, thermo, traj, verbose):
    ''' performs monte carlo in parallel for all configurations to generate new samples '''
    npress, ntemp = Pf.shape
    # list of delayed operations
    operations = [delayed(getSample)(x[i, j], v[i, j], box[i, j], name, el, units, lat, sz, mass, P[i], dt[i, j],
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
    # return lammps object, tries/acceptation counts, and mc params
    res = (natoms, x, v, temp, pe, ke, virial, box, vol,
           ntrypos, naccpos, ntryvol, naccvol, ntryhmc,
           nacchmc, accpos, accvol, acchmc, dpos, dbox, dt)
    return res


def getSamples(x, v, box, name, el, units, lat, sz, mass, P, dt,
               Et, Pf, ppos, pvol, phmc,
               ntrypos, naccpos,
               ntryvol, naccvol,
               ntryhmc, nacchmc,
               accpos, accvol, acchmc, dpos, dbox,
               T, mod, thermo, traj, verbose):
    ''' performs monte carlo for all configurations to generate new samples '''
    npress, ntemp = Pf.shape
    # loop through pressures
    if verbose:
        print('performing monte carlo')
    for i in xrange(npress):
        # loop through temperatures
        for j in xrange(ntemp):
            # get new sample configuration for press/temp combo
            dat = getSample(x[i, j], v[i, j], box[i, j], name, el, units, lat, sz, mass, P[i], dt[i, j],
                            Et[i, j], Pf[i, j], ppos, pvol, phmc,
                            ntrypos[i, j], naccpos[i, j],
                            ntryvol[i, j], naccvol[i, j],
                            ntryhmc[i, j], nacchmc[i, j],
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
    if verbose:
        print('writing data')
    for i in xrange(npress):
        for j in xrange(ntemp):
            writeThermo(thermo[i, j], temp[i, j], pe[i, j], ke[i, j], virial[i, j], vol[i, j],
                        accpos[i, j], accvol[i, j], acchmc[i, j])
            writeTraj(traj[i, j], natoms[i, j], box[i, j], x[i, j])
    # return lammps object, tries/acceptation counts, and mc params
    res = (natoms, x, v, temp, pe, ke, virial, box, vol,
           ntrypos, naccpos, ntryvol, naccvol, ntryhmc, nacchmc,
           accpos, accvol, acchmc, dpos, dbox, dt)
    return res

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
    res = natoms, x, v, temp, pe, ke, virial, box, vol
    return res

# -------------
# allocate data
# -------------

# thermo constants
ET = np.zeros((NPRESS, NTEMP), dtype=float)
PF = np.zeros((NPRESS, NTEMP), dtype=float)
# lammps objects and data storage files
THERMO = np.empty((NPRESS, NTEMP), dtype=object)
TRAJ = np.empty((NPRESS, NTEMP), dtype=object)
# system properties
natoms = np.zeros((NPRESS, NTEMP), dtype=int)
x = np.empty((NPRESS, NTEMP), dtype=object)
v = np.empty((NPRESS, NTEMP), dtype=object)
temp = np.zeros((NPRESS, NTEMP), dtype=float)
pe = np.zeros((NPRESS, NTEMP), dtype=float)
ke = np.zeros((NPRESS, NTEMP), dtype=float)
virial = np.zeros((NPRESS, NTEMP), dtype=float)
box = np.zeros((NPRESS, NTEMP), dtype=float)
vol = np.zeros((NPRESS, NTEMP), dtype=float)
# monte carlo tries/acceptations
ntrypos = np.zeros((NPRESS, NTEMP), dtype=float)
naccpos = np.zeros((NPRESS, NTEMP), dtype=float)
ntryvol = np.zeros((NPRESS, NTEMP), dtype=float)
naccvol = np.zeros((NPRESS, NTEMP), dtype=float)
ntryhmc = np.zeros((NPRESS, NTEMP), dtype=float)
nacchmc = np.zeros((NPRESS, NTEMP), dtype=float)
accpos = np.zeros((NPRESS, NTEMP), dtype=float)
accvol = np.zeros((NPRESS, NTEMP), dtype=float)
acchmc = np.zeros((NPRESS, NTEMP), dtype=float)
# max box adjustment
dbox = 0.03125*LAT[EL][1]*np.ones((NPRESS, NTEMP))
# max pos adjustment
dpos = 0.03125*LAT[EL][1]*np.ones((NPRESS, NTEMP))
# hmc timestep
dt = TIMESTEP[UNITS[EL]]*np.ones((NPRESS, NTEMP))

# -----------------
# initialize client
# -----------------

if PARALLEL:
    if DISTRIBUTED:
        # construct distributed cluster
        cluster = PBSCluster(queue=QUEUE, project=ALLOC, resource_spec='nodes=%d:ppn=%d' % (NODES, PPN), walltime='%d:00:00' % WALLTIME,
                             processes=NWORKER, cores=NTHREAD*NWORKER, memory=str(MEM)+'GB')
        cluster.start_workers(1)
        # start client with distributed cluster
        client = Client(cluster)
        while 'processes=0 cores=0' in str(client.scheduler_info):
            time.sleep(5)
            if VERBOSE:
                print(client.scheduler_info)
    else:
        # construct local cluster
        cluster = LocalCluster(n_workers=NWORKER, threads_per_worker=NTHREAD)
        # start client with local cluster
        client = Client(cluster)
        # display client information
        if VERBOSE:
            print(client.scheduler_info)

# ------------------
# initialize samples
# ------------------

if PARALLEL:
    operations = [delayed(defineConstants)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
    futures = client.compute(operations)
    if VERBOSE:
        print('setting constants')
        progress(futures)
    results = client.gather(futures)
    k = 0
    for i in xrange(NPRESS):
        for j in xrange(NTEMP):
            ET[i, j], PF[i, j] = results[k]
            k += 1
    operations = [delayed(sampleInit)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
    futures = client.compute(operations)
    if VERBOSE:
        print('\ninitializing samples')
        progress(futures)
    results = client.gather(futures)
    k = 0
    for i in xrange(NPRESS):
        for j in xrange(NTEMP):
            dat = results[k]
            k += 1
            natoms[i, j], x[i, j], v[i, j] = dat[:3]
            temp[i, j], pe[i, j], ke[i, j], virial[i, j], box[i, j], vol[i, j] = dat[3:9]
            THERMO[i, j], TRAJ[i, j] = dat[9:11]
    operations = [delayed(thermoHeader)(THERMO[i, j], NSMPL, CUTOFF, MOD, NSWPS, PPOS, PVOL, PHMC,
                                        NSTPS, SEED, EL, UNITS[EL], LAT[EL], SZ, MASS[EL], P[i], T[j],
                                        dt[i, j], dpos[i, j], dbox[i, j]) for i in xrange(NPRESS) for j in xrange(NTEMP)]
    futures = client.compute(operations)
    if VERBOSE:
        print('\nwriting thermo headers')
        progress(futures)
    del futures
    del results
else:
    # loop through pressures
    for i in xrange(NPRESS):
        # loop through temperatures
        for j in xrange(NTEMP):
            # set thermo constants
            ET[i, j], PF[i, j] = defineConstants(P[i], T[j])
            # initialize lammps object and data storage files
            dat = sampleInit(i, j, NAME, EL, UNITS[EL], LAT[EL], SZ, MASS[EL], P[i], dpos[i, j], dt[i, j])
            natoms[i, j], x[i, j], v[i, j] = dat[:3]
            temp[i, j], pe[i, j], ke[i, j], virial[i, j], box[i, j], vol[i, j] = dat[3:9]
            THERMO[i, j], THERMO[i, j] = dat[9:11]
            # write thermo file header
            thermoHeader(THERMO[i, j], NSMPL, CUTOFF, MOD, NSWPS, PPOS, PVOL, PHMC,
                         NSTPS, SEED, EL, UNITS[EL], LAT[EL], SZ, MASS[EL],
                         P[i], T[j], dt[i, j], dpos[i, j], dbox[i, j])

# -----------
# monte carlo
# -----------

# loop through to number of samples that need to be collected
for i in xrange(NSMPL):
    if VERBOSE:
        print('step:', i)
    # collect samples for all configurations
    if PARALLEL:
        dat = getSamplesPar(x, v, box, NAME, EL, UNITS[EL], LAT[EL], SZ, MASS[EL], P, dt,
                            ET, PF, PPOS, PVOL, PHMC,
                            ntrypos, naccpos,
                            ntryvol, naccvol,
                            ntryhmc, nacchmc,
                            accpos, accvol, acchmc, dpos, dbox,
                            T, MOD, THERMO, TRAJ, VERBOSE)
    else:
        dat = getSamples(x, v, box, NAME, EL, UNITS[EL], LAT[EL], SZ, MASS[EL], P, dt,
                         ET, PF, PPOS, PVOL, PHMC,
                         ntrypos, naccpos,
                         ntryvol, naccvol,
                         ntryhmc, nacchmc,
                         accpos, accvol, acchmc, dpos, dbox,
                         T, MOD, THERMO, TRAJ, VERBOSE)
    # update system data
    natoms[:, :], x[:, :], v[:, :] = dat[:3]
    temp[:, :], pe[:, :], ke[:, :], virial[:, :], box[:, :], vol[:, :] = dat[3:9]
    ntrypos[:, :], naccpos[:, :] = dat[9:11]
    ntryvol[:, :], naccvol[:, :] = dat[11:13]
    ntryhmc[:, :], nacchmc[:, :] = dat[13:15]
    accpos[:, :], accvol[:, :], acchmc[:, :] = dat[15:18]
    dpos[:, :], dbox[:, :], dt[:, :] = dat[18:21]
    # perform replica exchange markov chain monte carlo (parallel tempering)
    dat = repExch(natoms, x, v, temp, pe, ke, virial, box, vol, Et, Pf, VERBOSE)
    natoms[:, :], x[:, :], v[:, :] = dat[:3]
    temp[:, :], pe[:, :], ke[:, :], virial[:, :], box[:, :], vol[:, :] = dat[3:9]
if PARALLEL:
    # terminate client after completion
    client.close()

# ------------------
# final data storage
# ------------------

if VERBOSE:
    print('consolidating files')
# loop through pressures
for i in xrange(NPRESS):
    # get prefix
    prefix = fpref(NAME, EL, lat[EL], P[i])
    # open collected thermo data file
    with open(prefix+'.thrm', 'wb') as fo:
        # write data to collected thermo file
        for j in xrange(NTEMP):
            with open(thermo[i, j], 'r') as fi:
                k = 0
                for line in fi:
                    if '#' in line:
                        fo.write(line)
                    else:
                        k += 1
                        if k > CUTOFF:
                            fo.write(line)
    # open collected traj data file
    with open(prefix+'.traj', 'wb') as fo:
        # write data to collected traj file
        for j in xrange(NTEMP):
            with open(traj[i, j], 'r') as fi:
                k = 0
                for line in fi:
                    k += 1
                    if k > (natoms[i, j]+1)*CUTOFF:
                        fo.write(line)

# --------------
# clean up files
# --------------

if VERBOSE:
    print('cleaning files')
# remove all temporary files
for i in xrange(NPRESS):
    for j in xrange(NTEMP):
        os.remove(thermo[i, j])
        os.remove(traj[i, j])
