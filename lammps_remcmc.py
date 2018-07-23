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
PARSER = argparse.ArgumentParser()
PARSER.add_argument('-v', '--verbose', help='verbose output', action='store_true')
PARSER.add_argument('-p', '--parallel', help='parallel run', action='store_true')
PARSER.add_argument('-d', '--distributed', help='distributed run', action='store_true')
PARSER.add_argument('-q', '--queue', help='submission queue',
                    type=str, default='lasigma')
PARSER.add_argument('-a', '--allocation', help='submission allocation',
                    type=str, default='hpc_lasigma01')
PARSER.add_argument('-nn', '--nodes', help='number of nodes',
                    type=int, default=1)
PARSER.add_argument('-np', '--procs_per_node', help='number of processors per node',
                    type=int, default=16)
PARSER.add_argument('-w', '--walltime', help='job walltime',
                    type=int, default=24)
PARSER.add_argument('-m', '--memory', help='total job memory',
                    type=int, default=32)
PARSER.add_argument('-nw', '--workers', help='total job worker count',
                    type=int, default=16)
PARSER.add_argument('-nt', '--threads', help='threads per worker',
                    type=int, default=1)
PARSER.add_argument('-n', '--name', help='name of simulation',
                    type=str, default='test')
PARSER.add_argument('-e', '--element', help='element choice',
                    type=str, default='LJ')
PARSER.add_argument('-ss', '--supercell_size', help='supercell size',
                    type=int, default=4)
PARSER.add_argument('-pn', '--pressure_number', help='number of pressures',
                    type=int, default=4)
PARSER.add_argument('-pr', '--pressure_range', help='pressure range',
                    type=float, nargs=2, default=[2, 8])
PARSER.add_argument('-tn', '--temperature_number', help='number of temperatures',
                    type=int, default=48)
PARSER.add_argument('-tr', '--temperature_range', help='temperature range',
                    type=float, nargs=2, default=[0.25, 2.5])
PARSER.add_argument('-sc', '--sample_cutoff', help='sample cutoff',
                    type=int, default=1024)
PARSER.add_argument('-sn', '--sample_number', help='sample number',
                    type=int, default=1024)
PARSER.add_argument('-sm', '--sample_mod', help='sample record modulo',
                    type=int, default=128)
PARSER.add_argument('-pm', '--position_move', help='position monte carlo move probability',
                    type=float, default=0.015625)
PARSER.add_argument('-vm', '--volume_move', help='volume monte carlo move probability',
                    type=float, default=0.25)
PARSER.add_argument('-t', '--timesteps', help='hamiltonian monte carlo timesteps',
                    type=int, default=8)
# parse arguments
ARGS = PARSER.parse_args()

# booleans for run type
VERBOSE = ARGS.verbose
PARALLEL = ARGS.parallel
DISTRIBUTED = ARGS.distributed
# arguments for distributed run using pbs
QUEUE = ARGS.queue
ALLOC = ARGS.allocation
NODES = ARGS.nodes
PPN = ARGS.procs_per_node
WALLTIME = ARGS.walltime
MEM = ARGS.memory
# arguments for parallel run
NWORKER = ARGS.workers
NTHREAD = ARGS.threads
# simulation name
NAME = ARGS.name
# element choice
EL = ARGS.element
# supercell size in lattice parameters
SZ = ARGS.supercell_size
# pressure parameters
NPRESS = ARGS.pressure_number
LPRESS, HPRESS = ARGS.pressure_range
# temperature parameters
NTEMP = ARGS.temperature_number
LTEMP, HTEMP = ARGS.temperature_range
# sample cutoff
CUTOFF = ARGS.sample_cutoff
# number of recording samples
NSMPL = ARGS.sample_number
# record frequency
MOD = ARGS.sample_mod
# monte carlo probabilities
PPOS = ARGS.position_move
PVOL = ARGS.volume_move
# number of hamiltonian monte carlo timesteps
NSTPS = ARGS.timesteps

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
P = np.linspace(LPRESS, HPRESS, NPRESS, dtype=np.float32)
# temperature
T = np.linspace(LTEMP, HTEMP, NTEMP, dtype=np.float32)
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

# -------------
# allocate data
# -------------

# thermo constants
ET = np.zeros((NPRESS, NTEMP), dtype=np.float32)
PF = np.zeros((NPRESS, NTEMP), dtype=np.float32)
# lammps objects and data storage files
THERMO = np.empty((NPRESS, NTEMP), dtype=object)
TRAJ = np.empty((NPRESS, NTEMP), dtype=object)
# system properties
natoms = np.zeros((NPRESS, NTEMP), dtype=np.uint16)
x = np.empty((NPRESS, NTEMP), dtype=object)
v = np.empty((NPRESS, NTEMP), dtype=object)
temp = np.zeros((NPRESS, NTEMP), dtype=np.float32)
pe = np.zeros((NPRESS, NTEMP), dtype=np.float32)
ke = np.zeros((NPRESS, NTEMP), dtype=np.float32)
virial = np.zeros((NPRESS, NTEMP), dtype=np.float32)
box = np.zeros((NPRESS, NTEMP), dtype=np.float32)
vol = np.zeros((NPRESS, NTEMP), dtype=np.float32)
# monte carlo tries/acceptations
ntrypos = np.zeros((NPRESS, NTEMP), dtype=np.float32)
naccpos = np.zeros((NPRESS, NTEMP), dtype=np.float32)
ntryvol = np.zeros((NPRESS, NTEMP), dtype=np.float32)
naccvol = np.zeros((NPRESS, NTEMP), dtype=np.float32)
ntryhmc = np.zeros((NPRESS, NTEMP), dtype=np.float32)
nacchmc = np.zeros((NPRESS, NTEMP), dtype=np.float32)
accpos = np.zeros((NPRESS, NTEMP), dtype=np.float32)
accvol = np.zeros((NPRESS, NTEMP), dtype=np.float32)
acchmc = np.zeros((NPRESS, NTEMP), dtype=np.float32)
# max box adjustment
dbox = 0.03125*LAT[EL][1]*np.ones((NPRESS, NTEMP), dtype=np.float32)
# max pos adjustment
dpos = 0.03125*LAT[EL][1]*np.ones((NPRESS, NTEMP), dtype=np.float32)
# hmc timestep
dt = TIMESTEP[UNITS[EL]]*np.ones((NPRESS, NTEMP), dtype=np.float32)

# -----------------
# initialize client
# -----------------

if PARALLEL:
    if DISTRIBUTED:
        # construct distributed cluster
        CLUSTER = PBSCluster(queue=QUEUE, project=ALLOC,
                             resource_spec='nodes=%d:ppn=%d' % (NODES, PPN),
                             walltime='%d:00:00' % WALLTIME,
                             processes=NWORKER, cores=NTHREAD*NWORKER,
                             memory=str(MEM)+'GB')
        CLUSTER.start_workers(1)
        # start client with distributed cluster
        CLIENT = Client(CLUSTER)
        while 'processes=0 cores=0' in str(CLIENT.scheduler_info):
            time.sleep(5)
            if VERBOSE:
                print(CLIENT.scheduler_info)
    else:
        # construct local cluster
        CLUSTER = LocalCluster(n_workers=NWORKER, threads_per_worker=NTHREAD)
        # start client with local cluster
        CLIENT = Client(CLUSTER)
        # display client information
        if VERBOSE:
            print(CLIENT.scheduler_info)

# ----------------
# unit definitions
# ----------------

def define_constants(i, j):
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
        KB = 1.0                                       # boltzmann constant (unitless)
        ET = KB*T[j]                                   # thermal energy [T*]
        PF = P[i]/(KB*T[j])                            # metropolis prefactor [1/r*^3]
    return ET, PF

# -----------------------------
# output file utility functions
# -----------------------------


def thermo_header(i, j):
    ''' writes header containing simulation information to thermo file '''
    with open(THERMO[i, j], 'wb') as fo:
        fo.write('#----------------------\n')
        fo.write('# simulation parameters\n')
        fo.write('#----------------------\n')
        fo.write('# nsmpl:    %d\n' % NSMPL)
        fo.write('# cutoff:   %d\n' % CUTOFF)
        fo.write('# mod:      %d\n' % MOD)
        fo.write('# nswps:    %d\n' % NSWPS)
        fo.write('# ppos:     %f\n' % PPOS)
        fo.write('# pvol:     %f\n' % PVOL)
        fo.write('# phmc:     %f\n' % PHMC)
        fo.write('# nstps:    %d\n' % NSTPS)
        fo.write('# seed:     %d\n' % SEED)
        fo.write('#----------------------\n')
        fo.write('# material properties\n')
        fo.write('#----------------------\n')
        fo.write('# element:  %s\n' % EL)
        fo.write('# units:    %s\n' % UNITS[EL])
        fo.write('# lattice:  %s\n' % LAT[EL][0])
        fo.write('# latpar:   %f\n' % LAT[EL][1])
        fo.write('# size:     %d\n' % SZ)
        fo.write('# mass:     %f\n' % MASS[EL])
        fo.write('# press:    %f\n' % P[i])
        fo.write('# temp:     %f\n' % T[j])
        fo.write('# dposmax:  %f\n' % dpos[i, j])
        fo.write('# dboxmax:  %f\n' % dbox[i, j])
        fo.write('# timestep: %f\n' % dt[i, j])
        fo.write('# ------------------------------------------------------------\n')
        fo.write('# | temp | pe | ke | virial | vol | accpos | accvol | acchmc |\n')
        fo.write('# ------------------------------------------------------------\n')


def write_thermo(i, j):
    ''' writes thermodynamic properties to thermo file '''
    # write data to file
    therm_args = (temp[i, j], pe[i, j], ke[i, j],
                  virial[i, j], vol[i, j],
                  accpos[i, j], accvol[i, j], acchmc[i, j])
    with open(THERMO[i, j], 'ab') as fo:
        fo.write('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n' % therm_args)


def write_traj(i, j):
    ''' writes trajectory data to traj file '''
    # write data to file
    with open(TRAJ[i, j], 'ab') as fo:
        fo.write('%d %.4E\n' % (natoms[i, j], box[i, j]))
        for k in xrange(natoms[i, j]):
            fo.write('%.4E %.4E %.4E\n' % tuple(x[i, j][3*k:3*k+3]))


def write_output():
    ''' writes all sample outputs for a monte carlo step '''
    if PARALLEL:
        # write to data storage files
        operations = [delayed(write_thermo)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('\nwriting thermo data')
            progress(futures)
        operations = [delayed(write_traj)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('\nwriting traj data')
            progress(futures)
    else:
        # write to data storage files
        if VERBOSE:
            print('writing data')
        for i in xrange(NPRESS):
            for j in xrange(NTEMP):
                write_thermo(i, j)
                write_traj(i, j)

    # write to data storage files
    # if VERBOSE:
        # if PARALLEL:
            # print('\nwriting data')
        # else:
            # print('writing data')
    # for i in xrange(NPRESS):
        # for j in xrange(NTEMP):
            # write_thermo(i, j)
            # write_traj(i, j)

# ---------------------------------
# lammps file/object initialization
# ---------------------------------


def file_prefix(i):
    ''' returns file prefix for simulation '''
    prefix = '%s.%s.%s.%d.lammps' % (NAME, EL.lower(), LAT[EL][0], int(P[i]))
    return os.getcwd()+'/'+prefix


def lammps_input(i, j):
    ''' constructs input file for lammps
        takes element name, lattice definitions, size, and simulation name
        returns input file name '''
    lj_param = (1.0, 1.0)
    # convert lattice definition list to strings
    prefix = file_prefix(i)
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
        pc_pref = 'pair_coeff * *'
        if EL == 'Ti':
            fo.write('pair_style meam/c\n')
            fo.write('mass 1 47.867\n')
            fo.write('%s library.meam Ti Al TiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % (pc_pref, EL))
        if EL == 'Al':
            fo.write('pair_style meam/c\n')
            fo.write('mass 1 %f\n' % MASS[EL])
            fo.write('%s library.meam Ti Al TiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % (pc_pref, EL))
        if EL == 'Ni':
            fo.write('pair_style meam/c\n')
            fo.write('mass 1 %f\n' % MASS[EL])
            fo.write('%s library.Ni.meam Ni Ni.meam %s\n\n' % (pc_pref, EL))
        if EL == 'Cu':
            fo.write('pair_style meam/c\n')
            fo.write('mass 1 %f\n' % MASS[EL])
            fo.write('%s library.Cu.meam Cu Cu.meam %s\n\n' % (pc_pref, EL))
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


def lammps_extract(lmps):
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


def sample_init(i, j):
    ''' initializes system info and data storage files '''
    # generate input file
    lmpsfilein = lammps_input(i, j)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfilein)
    # minimize lattice structure
    lmps.command('unfix 1')
    lmps.command('fix 1 all box/relax iso %f vmax %f' % (P[i], 0.0009765625))
    lmps.command('minimize 0.0 %f %d %d' % (1.49011612e-8, 1024, 8192))
    seed = np.random.randint(1, 2**16)
    lmps.command('displace_atoms all random %f %f %f %d' % (3*(dpos[i, j],)+(seed,)))
    # extract all system info
    natomsinit, xinit, vinit, tempinit, peinit, keinit, virialinit, boxinit, volinit = lammps_extract(lmps)
    # data storage files
    thermoinit = lmpsfilein.replace('.in', '%02d%02d.thrm' % (i, j))
    trajinit = lmpsfilein.replace('.in', '%02d%02d.traj' % (i, j))
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
    res = (natomsinit, xinit, vinit, tempinit, peinit, keinit, virialinit, boxinit, volinit,
           thermoinit, trajinit)
    return res

def samples_init():
    ''' initializes all samples '''
    if PARALLEL:
        operations = [delayed(define_constants)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('setting constants')
            progress(futures)
        results = CLIENT.gather(futures)
        k = 0
        for i in xrange(NPRESS):
            for j in xrange(NTEMP):
                ET[i, j], PF[i, j] = results[k]
                k += 1
        operations = [delayed(sample_init)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('\ninitializing samples')
            progress(futures)
        results = CLIENT.gather(futures)
        k = 0
        for i in xrange(NPRESS):
            for j in xrange(NTEMP):
                dat = results[k]
                k += 1
                natoms[i, j], x[i, j], v[i, j] = dat[:3]
                temp[i, j], pe[i, j], ke[i, j], virial[i, j], box[i, j], vol[i, j] = dat[3:9]
                THERMO[i, j], TRAJ[i, j] = dat[9:11]
        operations = [delayed(thermo_header)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('\nwriting thermo headers')
            progress(futures)
            print('\n')
    else:
        ('initializing constants, samples, and output files')
        # loop through pressures
        for i in xrange(NPRESS):
            # loop through temperatures
            for j in xrange(NTEMP):
                # set thermo constants
                ET[i, j], PF[i, j] = define_constants(i, j)
                # initialize lammps object and data storage files
                dat = sample_init(i, j)
                natoms[i, j], x[i, j], v[i, j] = dat[:3]
                temp[i, j], pe[i, j], ke[i, j], virial[i, j], box[i, j], vol[i, j] = dat[3:9]
                THERMO[i, j], TRAJ[i, j] = dat[9:11]
                # write thermo file header
                thermo_header(i, j)


def lammps_init(i, j):
    ''' initializes system info and data storage files '''
    # generate input file
    lmpsfilein = lammps_input(i, j)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfilein)
    # set system info
    box_dim = (3*(box[i, j],))
    lmps.command('change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box' % box_dim)
    lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x[i, j]))
    lmps.scatter_atoms('v', 1, 3, np.ctypeslib.as_ctypes(v[i, j]))
    lmps.command('run 0')
    return lmps

# -----------------
# monte carlo moves
# -----------------


def bulk_position_mc(i, j, lmps, ntryposnew, naccposnew):
    ''' classic position monte carlo
        simultaneous random displacement of atoms
        accepts/rejects based on energy metropolis criterion '''
    ntryposnew += 1
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    pe = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
    seed = np.random.randint(1, 2**16)
    lmps.command('displace_atoms all random %f %f %f %d' % (3*(dpos[i, j],)+(seed,)))
    lmps.command('run 0')
    xnew = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    penew = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
    dE = penew-pe
    if np.random.rand() <= np.min([1, np.exp(-dE)]):
        # update pos acceptations
        naccposnew += 1
        # save new physical properties
        x = xnew
        pe = penew
    else:
        # revert physical properties
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
        lmps.command('run 0')
    # return lammps object and tries/acceptations
    res = lmps, ntryposnew, naccposnew
    return res


def iter_position_mc(i, j, lmps, ntryposnew, naccposnew):
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
        ntryposnew += 1
        # save current physical properties
        x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
        pe = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
        xnew = np.copy(x)
        xnew[3*k:3*k+3] += (np.random.rand(3)-0.5)*dpos
        xnew[3*k:3*k+3] -= np.floor(xnew[3*k:3*k+3]/box)*box
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
        lmps.command('run 0')
        penew = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
        dE = penew-pe
        if np.random.rand() <= np.min([1, np.exp(-dE)]):
            # update pos acceptations
            naccposnew += 1
            # save new physical properties
            x = xnew
            pe = penew
        else:
            # revert physical properties
            lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
            lmps.command('run 0')
    # return lammps object and tries/acceptations
    res = lmps, ntryposnew, naccposnew
    return res


def volume_mc(i, j, lmps, ntryvolnew, naccvolnew):
    ''' isobaric-isothermal volume monte carlo
        scales box and positions
        accepts/rejects based on enthalpy metropolis criterion '''
    # update volume tries
    ntryvolnew += 1
    # save current physical properties
    natoms = lmps.extract_global('natoms', 0)
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    vol = np.power(box, 3)
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    pe = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
    # save new physical properties
    boxnew = box+(np.random.rand()-0.5)*dbox[i, j]
    volnew = np.power(boxnew, 3)
    scalef = boxnew/box
    xnew = scalef*x
    # apply new physical properties
    lmps.command('change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box' % (3*(boxnew,)))
    lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
    lmps.command('run 0')
    penew = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
    # calculate enthalpy criterion
    dH = (penew-pe)+PF[i, j]*(volnew-vol)-natoms*np.log(volnew/vol)
    if np.random.rand() <= np.min([1, np.exp(-dH)]):
        # update volume acceptations
        naccvolnew += 1
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
    res = lmps, ntryvolnew, naccvolnew
    return res


def hamiltonian_mc(i, j, lmps, ntryhmcnew, nacchmcnew):
    ''' hamiltionian monte carlo
        short md run at generated velocities for desired temp
        accepts/rejects based on energy metropolis criterion '''
    # update hmc tries
    ntryhmcnew += 1
    # set new atom velocities and initialize
    seed = np.random.randint(1, 2**16)
    lmps.command('velocity all create %f %d dist gaussian' % (T[j], seed))
    lmps.command('velocity all zero linear')
    lmps.command('velocity all zero angular')
    lmps.command('timestep %f' % dt[i, j])
    lmps.command('run 0')
    # save current physical properties
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    v = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3)))
    pe = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
    ke = lmps.extract_compute('thermo_ke', None, 0)/ET[i, j]
    etot = pe+ke
    # run md
    lmps.command('run %d' % NSTPS)  # this part should be implemented as parallel
    # set new physical properties
    xnew = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    vnew = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3)))
    penew = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
    kenew = lmps.extract_compute('thermo_ke', None, 0)/ET[i, j]
    etotnew = penew+kenew
    # calculate hamiltonian criterion
    dE = etotnew-etot
    if np.random.rand() <= np.min([1, np.exp(-dE)]):
        # update hamiltonian acceptations
        nacchmcnew += 1
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
    res = lmps, ntryhmcnew, nacchmcnew
    return res

# ----------------------------
# monte carlo parameter update
# ----------------------------


def update_mc_param(i, j):
    ''' adaptive update of monte carlo parameters '''
    # update position displacment for pos-mc
    if accpos[i, j] < 0.5:
        dposnew = 0.9375*dpos[i, j]
    else:
        dposnew = 1.0625*dpos[i, j]
    # update box displacement for vol-mc
    if accvol[i, j] < 0.5:
        dboxnew = 0.9375*dbox[i, j]
    else:
        dboxnew = 1.0625*dbox[i, j]
    # update timestep for hmc
    if acchmc[i, j] < 0.5:
        dtnew = 0.9375*dt[i, j]
    else:
        dtnew = 1.0625*dt[i, j]
    # return new mc params
    res = dposnew, dboxnew, dtnew
    return res

# ---------------------
# monte carlo procedure
# ---------------------


def move_mc(i, j, lmps, ntryposnew, naccposnew, ntryvolnew, naccvolnew, ntryhmcnew, nacchmcnew):
    ''' performs monte carlo moves '''
    roll = np.random.rand()
    # position monte carlo
    if roll <= PPOS:
        lmps, ntryposnew, naccposnew = bulk_position_mc(i, j, lmps, ntryposnew, naccposnew)
        # lmps, ntryposnew, naccposnew = iter_position_mc(i, j, lmps, ntryposnew, naccposnew)
    # volume monte carlo
    elif roll <= (PPOS+PVOL):
        lmps, ntryvolnew, naccvolnew = volume_mc(i, j, lmps, ntryvolnew, naccvolnew)
    # hamiltonian monte carlo
    else:
        lmps, ntryhmcnew, nacchmcnew = hamiltonian_mc(i, j, lmps, ntryhmcnew, nacchmcnew)
    res = lmps, ntryposnew, naccposnew, ntryvolnew, naccvolnew, ntryhmcnew, nacchmcnew
    return res


def get_sample(i, j):
    ''' performs enough monte carlo moves to generate a sample (determined by mod) '''
    # initialize lammps object
    lmps = lammps_init(i, j)
    ntryposnew, naccposnew, ntryvolnew, naccvolnew, ntryhmcnew, nacchmcnew = (ntrypos[i, j], naccpos[i, j],
                                                                              ntryvol[i, j], naccvol[i, j],
                                                                              ntryhmc[i, j], nacchmc[i, j])
    # loop through monte carlo moves
    for k in xrange(MOD):
        dat = move_mc(i, j, lmps,
                      ntryposnew, naccposnew,
                      ntryvolnew, naccvolnew,
                      ntryhmcnew, nacchmcnew)
        lmps = dat[0]
        ntryposnew, naccposnew = dat[1:3]
        ntryvolnew, naccvolnew = dat[3:5]
        ntryhmcnew, nacchmcnew = dat[5:7]
    # extract system properties
    natomsnew, xnew, vnew, tempnew, penew, kenew, virialnew, boxnew, volnew = lammps_extract(lmps)
    # close lammps and remove input file
    lmps.close()
    # acceptation ratios
    with np.errstate(invalid='ignore'):
        accposnew = np.nan_to_num(np.float64(naccposnew)/np.float64(ntryposnew))
        accvolnew = np.nan_to_num(np.float64(naccvolnew)/np.float64(ntryvolnew))
        acchmcnew = np.nan_to_num(np.float64(nacchmcnew)/np.float64(ntryhmcnew))
    # update mc params
    dposnew, dboxnew, dtnew = update_mc_param(i, j)
    # return lammps object, tries/acceptation counts, and mc params
    res = (natomsnew, xnew, vnew, tempnew, penew, kenew, virialnew, boxnew, volnew,
           ntryposnew, naccposnew, ntryvolnew, naccvolnew, ntryhmcnew, nacchmcnew,
           accposnew, accvolnew, acchmcnew, dposnew, dboxnew, dtnew)
    return res


def get_samples():
    ''' performs monte carlo for all configurations to generate new samples '''
    if PARALLEL:
        # list of delayed operations
        operations = [delayed(get_sample)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        # submit futures to client
        futures = CLIENT.compute(operations)
        # progress bar
        if VERBOSE:
            print('performing monte carlo')
            progress(futures)
        # gather results from workers
        results = CLIENT.gather(futures)
        # update system properties and monte carlo parameters
        k = 0
        for i in xrange(NPRESS):
            for j in xrange(NTEMP):
                dat = results[k]
                k += 1
                natoms[i, j], x[i, j], v[i, j] = dat[:3]
                temp[i, j], pe[i, j], ke[i, j], virial[i, j], box[i, j], vol[i, j] = dat[3:9]
                ntrypos[i, j], naccpos[i, j] = dat[9:11]
                ntryvol[i, j], naccvol[i, j] = dat[11:13]
                ntryhmc[i, j], nacchmc[i, j] = dat[13:15]
                accpos[i, j], accvol[i, j], acchmc[i, j] = dat[15:18]
                dpos[i, j], dbox[i, j], dt[i, j] = dat[18:21]
    else:
        # loop through pressures
        if VERBOSE:
            print('performing monte carlo')
        for i in xrange(NPRESS):
            # loop through temperatures
            for j in xrange(NTEMP):
                # get new sample configuration for press/temp combo
                dat = get_sample(i, j)
                natoms[i, j], x[i, j], v[i, j] = dat[:3]
                temp[i, j], pe[i, j], ke[i, j], virial[i, j], box[i, j], vol[i, j] = dat[3:9]
                ntrypos[i, j], naccpos[i, j] = dat[9:11]
                ntryvol[i, j], naccvol[i, j] = dat[11:13]
                ntryhmc[i, j], nacchmc[i, j] = dat[13:15]
                accpos[i, j], accvol[i, j], acchmc[i, j] = dat[15:18]
                dpos[i, j], dbox[i, j], dt[i, j] = dat[18:21]
                dpos[i, j], dbox[i, j], dt[i, j] = dat[15:18]

# -----------------------------------------
# replica exchange markov chain monte carlo
# -----------------------------------------


def replica_exchange():
    ''' performs parallel tempering acrros all samples
        accepts/rejects based on enthalpy metropolis criterion '''
    # flatten system properties and constants
    natomsf = natoms.reshape(-1)
    xf = x.reshape(-1)
    vf = v.reshape(-1)
    tempf = temp.reshape(-1)
    pef = pe.reshape(-1)
    kef = ke.reshape(-1)
    virialf = virial.reshape(-1)
    boxf = box.reshape(-1)
    volf = vol.reshape(-1)
    ETf = ET.reshape(-1)
    PFf = PF.reshape(-1)
    # catalog swaps
    swaps = 0
    # loop through upper right triangular matrix
    for i in xrange(len(PFf)):
        for j in xrange(i+1, len(PFf)):
            # configuration energies
            etoti = pef[i]+kef[i]
            etotj = pef[j]+kef[j]
            # change in enthalpy
            dH = (etoti-etotj)*(1/ETf[i]-1/ETf[j])+(PFf[i]-PFf[j])*(volf[i]-volf[j])
            if np.random.rand() <= np.min([1, np.exp(dH)]):
                swaps += 1
                # swap lammps objects
                natomsf[j], natomsf[i] = natomsf[i], natomsf[j]
                xf[j], xf[i] = xf[i], xf[j]
                vf[j], vf[i] = vf[i], vf[j]
                tempf[j], tempf[i] = tempf[i], tempf[j]
                pef[j], pef[i] = pef[i], pef[j]
                kef[j], kef[i] = kef[i], kef[j]
                virialf[j], virialf[i] = virialf[i], virialf[j]
                boxf[j], boxf[i] = boxf[i], boxf[j]
                volf[j], volf[i] = volf[i], volf[j]
    if VERBOSE:
        # if PARALLEL:
            # print('\n%d replica exchanges performed' % swaps)
        # else:
            # print('%d replica exchanges performed' % swaps)
        print('%d replica exchanges performed' % swaps)
    # reshape system properties and constants
    natoms[:, :] = natomsf.reshape((NPRESS, NTEMP))
    x[:, :] = xf.reshape((NPRESS, NTEMP))
    v[:, :] = vf.reshape((NPRESS, NTEMP))
    temp[:, :] = tempf.reshape((NPRESS, NTEMP))
    pe[:, :] = pef.reshape((NPRESS, NTEMP))
    ke[:, :] = kef.reshape((NPRESS, NTEMP))
    virial[:, :] = virialf.reshape((NPRESS, NTEMP))
    box[:, :] = boxf.reshape((NPRESS, NTEMP))
    vol[:, :] = volf.reshape((NPRESS, NTEMP))

# -----------
# monte carlo
# -----------

# initialize samples
samples_init()
# loop through to number of samples that need to be collected
for i in xrange(NSMPL):
    if VERBOSE:
        print('step: %d' % i)
    # collect samples for all configurations
    get_samples()
    # write to output files
    write_output()
    # perform replica exchange markov chain monte carlo (parallel tempering)
    replica_exchange()
    CLIENT.restart()
if PARALLEL:
    # terminate client after completion
    CLIENT.close()

# ------------------
# final data storage
# ------------------

if VERBOSE:
    print('consolidating files')
# loop through pressures
for i in xrange(NPRESS):
    # get prefix
    prefix = file_prefix(i)
    # open collected thermo data file
    with open(prefix+'.thrm', 'wb') as fo:
        # write data to collected thermo file
        for j in xrange(NTEMP):
            with open(THERMO[i, j], 'r') as fi:
                k = 0
                for line in fi:
                    if '#' in line:
                        fo.write(line)
                    else:
                        k += 1
                        # check against cutoff
                        if k > CUTOFF:
                            fo.write(line)
    # open collected traj data file
    with open(prefix+'.traj', 'wb') as fo:
        # write data to collected traj file
        for j in xrange(NTEMP):
            with open(TRAJ[i, j], 'r') as fi:
                k = 0
                for line in fi:
                    k += 1
                    # check against cutoff
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
        os.remove(THERMO[i, j])
        os.remove(TRAJ[i, j])
