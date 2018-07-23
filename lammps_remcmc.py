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
NATOMS = np.zeros((NPRESS, NTEMP), dtype=np.uint16)
X = np.empty((NPRESS, NTEMP), dtype=object)
V = np.empty((NPRESS, NTEMP), dtype=object)
TEMP = np.zeros((NPRESS, NTEMP), dtype=np.float32)
PE = np.zeros((NPRESS, NTEMP), dtype=np.float32)
KE = np.zeros((NPRESS, NTEMP), dtype=np.float32)
VIRIAL = np.zeros((NPRESS, NTEMP), dtype=np.float32)
BOX = np.zeros((NPRESS, NTEMP), dtype=np.float32)
VOL = np.zeros((NPRESS, NTEMP), dtype=np.float32)
# monte carlo tries/acceptations
NTP = np.zeros((NPRESS, NTEMP), dtype=np.float32)
NAP = np.zeros((NPRESS, NTEMP), dtype=np.float32)
NTV = np.zeros((NPRESS, NTEMP), dtype=np.float32)
NAV = np.zeros((NPRESS, NTEMP), dtype=np.float32)
NTH = np.zeros((NPRESS, NTEMP), dtype=np.float32)
NAH = np.zeros((NPRESS, NTEMP), dtype=np.float32)
AP = np.zeros((NPRESS, NTEMP), dtype=np.float32)
AV = np.zeros((NPRESS, NTEMP), dtype=np.float32)
AH = np.zeros((NPRESS, NTEMP), dtype=np.float32)
# max box adjustment
DL = 0.0078125*SZ*LAT[EL][1]*np.ones((NPRESS, NTEMP), dtype=np.float32)
# max pos adjustment
DX = 0.03125*LAT[EL][1]*np.ones((NPRESS, NTEMP), dtype=np.float32)
# hmc timestep
DT = TIMESTEP[UNITS[EL]]*np.ones((NPRESS, NTEMP), dtype=np.float32)


def sys_state(i, j):
    ''' returns system state '''
    system_properties = X[i, j], V[i, j], BOX[i, j]
    mc_counts = NTP[i, j], NAP[i, j], NTV[i, j], NAV[i, j], NTH[i, j], NAH[i, j]
    mc_move_params = DX[i, j], DL[i, j], DT[i, j]
    return system_properties+mc_counts+mc_move_params


def mc_state(i, j):
    ''' returns monte carlo parameter state '''
    return AP[i, j], AV[i, j], AH[i, j], DL[i, j], DX[i, j], DT[i, j]

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

def init_constant(i, j):
    ''' calculates thermodynamic constants for a sample '''
    if UNITS[EL] == 'real':
        na = 6.0221409e23                              # avagadro number [num/mol]
        kb = 3.29983e-27                               # boltzmann constant [kcal/K]
        r = kb*na                                      # gas constant [kcal/(mol K)]
        et = r*T[j]                                    # thermal energy [kcal/mol]
        pf = 1e-30*(1.01325e5*P[i])/(4.184e3*kb*T[j])  # metropolis prefactor [1/A^3]
    if UNITS[EL] == 'metal':
        kb = 8.61733e-5                                # boltzmann constant [eV/K]
        et = kb*T[i]                                   # thermal energy [eV]
        pf = 1e-30*(1e5*P[i])/(1.60218e-19*kb*T[j])    # metropolis prefactor [1/A^3]
    if UNITS[EL] == 'lj':
        kb = 1.0                                       # boltzmann constant (unitless)
        et = kb*T[j]                                   # thermal energy [T*]
        pf = P[i]/(kb*T[j])                            # metropolis prefactor [1/r*^3]
    return et, pf


def init_constants():
    ''' calculates thermodynamic constants for all samples '''
    if PARALLEL:
        operations = [delayed(init_constant)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('initializing constants')
            progress(futures)
            print('\n')
        results = CLIENT.gather(futures)
    else:
        results = [init_constant(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
    return results


def set_constants(results):
    ''' sets thermodynamic constants for all samples '''
    k = 0
    for i in xrange(NPRESS):
        for j in xrange(NTEMP):
            ET[i, j], PF[i, j] = results[k]
            k += 1

# -----------------------------
# output file utility functions
# -----------------------------


def file_prefix(i):
    ''' returns filename prefix for simulation '''
    prefix = '%s.%s.%s.%d.lammps' % (NAME, EL.lower(), LAT[EL][0], int(P[i]))
    return os.getcwd()+'/'+prefix


def init_output(i, j):
    ''' initializes output filenames for a sample '''
    thermo = file_prefix(i)+'%02d%02d.thrm' % (i, j)
    traj = thermo.replace('thrm', 'traj')
    try:
        os.remove(thermo)
    except:
        pass
    try:
        os.remove(traj)
    except:
        pass
    return thermo, traj


def init_outputs():
    ''' initializes output filenames for all samples '''
    if PARALLEL:
        operations = [delayed(init_output)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('initializing outputs')
            progress(futures)
            print('\n')
        results = CLIENT.gather(futures)
    else:
        if VERBOSE:
            print('initializing outputs')
        results = [init_output(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
    return results


def set_outputs(results):
    ''' sets output filenames '''
    k = 0
    for i in xrange(NPRESS):
        for j in xrange(NTEMP):
            THERMO[i, j], TRAJ[i, j] = results[k]
            k += 1


def init_header(i, j):
    ''' writes header for a sample '''
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
        fo.write('# dx:       %f\n' % DX[i, j])
        fo.write('# dl:       %f\n' % DL[i, j])
        fo.write('# dt:       %f\n' % DT[i, j])
        fo.write('# ------------------------------------------------------------\n')
        fo.write('# | temp | pe | ke | virial | vol | accpos | accvol | acchmc |\n')
        fo.write('# ------------------------------------------------------------\n')


def init_headers():
    ''' writes headers for all samples '''
    if PARALLEL:
        operations = [delayed(init_header)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('initializing headers')
            progress(futures)
            print('\n')
    else:
        if VERBOSE:
            print('initializing headers')
        for i in xrange(NPRESS):
            for j in xrange(NTEMP):
                init_header(i, j)


def write_thermo(i, j):
    ''' writes thermodynamic properties to thermo file '''
    therm_args = (TEMP[i, j], PE[i, j], KE[i, j], VIRIAL[i, j], VOL[i, j],
                  AP[i, j], AV[i, j], AH[i, j])
    with open(THERMO[i, j], 'ab') as fo:
        fo.write('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n' % therm_args)


def write_traj(i, j):
    ''' writes trajectory data to traj file '''
    with open(TRAJ[i, j], 'ab') as fo:
        fo.write('%d %.4E\n' % (NATOMS[i, j], BOX[i, j]))
        for k in xrange(NATOMS[i, j]):
            fo.write('%.4E %.4E %.4E\n' % tuple(X[i, j][3*k:3*k+3]))


def write_output(i, j):
    ''' writes output for a sample '''
    write_thermo(i, j)
    write_traj(i, j)


def write_outputs():
    ''' writes outputs for all samples '''
    if PARALLEL:
        operations = [delayed(write_output)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('writing outputs')
            progress(futures)
            print('\n')
    else:
        if VERBOSE:
            print('writing outputs')
        for i in xrange(NPRESS):
            for j in xrange(NTEMP):
                write_output(i, j)

# ---------------------------------
# lammps file/object initialization
# ---------------------------------


def lammps_input(i, j):
    ''' constructs input file for lammps '''
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
        fo.write('timestep %f\n' % DT[i, j])
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
    return natoms, x, v, temp, pe, ke, virial, box, vol


def init_sample(i, j):
    ''' initializes sample '''
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
    lmps.command('displace_atoms all random %f %f %f %d' % (3*(DX[i, j],)+(seed,)))
    # extract all system info
    natoms, x, v, temp, pe, ke, virial, box, vol = lammps_extract(lmps)
    lmps.close()
    # return system info and data storage files
    return natoms, x, v, temp, pe, ke, virial, box, vol


def init_samples():
    ''' initializes all samples '''
    if PARALLEL:
        operations = [delayed(init_sample)(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('initializing samples')
            progress(futures)
            print('\n')
        results = CLIENT.gather(futures)
    else:
        if VERBOSE:
            print('initializing samples')
        results = [init_sample(i, j) for i in xrange(NPRESS) for j in xrange(NTEMP)]
    return results


def set_samples(results):
    ''' sets all samples '''
    k = 0
    for i in xrange(NPRESS):
        for j in xrange(NTEMP):
            NATOMS[i, j], X[i, j], V[i, j] = results[k][:3]
            TEMP[i, j], PE[i, j], KE[i, j], VIRIAL[i, j], BOX[i, j], VOL[i, j] = results[k][3:9]
            k += 1


def init_lammps(i, j, x, v, box):
    ''' initializes lammps '''
    # generate input file
    lmpsfilein = lammps_input(i, j)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfilein)
    # set system info
    box_dim = (3*(box,))
    lmps.command('change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box' % box_dim)
    lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
    lmps.scatter_atoms('v', 1, 3, np.ctypeslib.as_ctypes(v))
    lmps.command('run 0')
    return lmps

# -----------------
# monte carlo moves
# -----------------


def bulk_position_mc(i, j, lmps, ntp, nap, dx):
    ''' classic position monte carlo (bulk) '''
    ntp += 1
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    pe = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
    seed = np.random.randint(1, 2**16)
    lmps.command('displace_atoms all random %f %f %f %d' % (3*(dx,)+(seed,)))
    lmps.command('run 0')
    xnew = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    penew = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
    dE = penew-pe
    if np.random.rand() <= np.min([1, np.exp(-dE)]):
        # update pos acceptations
        nap += 1
    else:
        # revert physical properties
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
        lmps.command('run 0')
    # return lammps object and tries/acceptations
    return lmps, ntp, nap


def iter_position_mc(i, j, lmps, ntp, nap, dx):
    ''' classic position monte carlo (iterative) '''
    # get number of atoms
    natoms = lmps.extract_global('natoms', 0)
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    # loop through atoms
    for k in xrange(natoms):
        # update position tries
        ntp += 1
        # save current physical properties
        x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
        pe = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
        xnew = np.copy(x)
        xnew[3*k:3*k+3] += (np.random.rand(3)-0.5)*dx
        xnew[3*k:3*k+3] -= np.floor(xnew[3*k:3*k+3]/box)*box
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
        lmps.command('run 0')
        penew = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
        dE = penew-pe
        if np.random.rand() <= np.min([1, np.exp(-dE)]):
            # update pos acceptations
            nap += 1
        else:
            # revert physical properties
            lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
            lmps.command('run 0')
    # return lammps object and tries/acceptations
    return lmps, ntp, nap


def volume_mc(i, j, lmps, ntv, nav, dl):
    ''' isobaric-isothermal volume monte carlo '''
    # update volume tries
    ntv += 1
    # save current physical properties
    natoms = lmps.extract_global('natoms', 0)
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    vol = np.power(box, 3)
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    pe = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
    # save new physical properties
    boxnew = box+(np.random.rand()-0.5)*dl
    volnew = np.power(boxnew, 3)
    scalef = boxnew/box
    xnew = scalef*x
    # apply new physical properties
    box_cmd = 'change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box'
    lmps.command(box_cmd % (3*(boxnew,)))
    lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
    lmps.command('run 0')
    penew = lmps.extract_compute('thermo_pe', None, 0)/ET[i, j]
    # calculate enthalpy criterion
    dH = (penew-pe)+PF[i, j]*(volnew-vol)-natoms*np.log(volnew/vol)
    if np.random.rand() <= np.min([1, np.exp(-dH)]):
        # update volume acceptations
        nav += 1
    else:
        # revert physical properties
        lmps.command(box_cmd % (3*(box,)))
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
        lmps.command('run 0')
    # return lammps object and tries/acceptations
    return lmps, ntv, nav


def hamiltonian_mc(i, j, lmps, nth, nah, dt):
    ''' hamiltionian monte carlo '''
    # update hmc tries
    nth += 1
    # set new atom velocities and initialize
    seed = np.random.randint(1, 2**16)
    lmps.command('velocity all create %f %d dist gaussian' % (T[j], seed))
    lmps.command('velocity all zero linear')
    lmps.command('velocity all zero angular')
    lmps.command('timestep %f' % dt)
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
        nah += 1
    else:
        # revert physical properties
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
        lmps.scatter_atoms('v', 1, 3, np.ctypeslib.as_ctypes(v))
        lmps.command('run 0')
    # return lammps object and tries/acceptations
    return lmps, nth, nah


def move_mc(i, j, lmps, ntp, nap, ntv, nav, nth, nah, dx, dl, dt):
    ''' performs monte carlo moves '''
    roll = np.random.rand()
    # position monte carlo
    if roll <= PPOS:
        lmps, ntp, nap = bulk_position_mc(i, j, lmps, ntp, nap, dx)
        # lmps, ntp, nap = iter_position_mc(i, j, lmps, ntp, nap, dx)
    # volume monte carlo
    elif roll <= (PPOS+PVOL):
        lmps, ntv, nav = volume_mc(i, j, lmps, ntv, nav, dl)
    # hamiltonian monte carlo
    else:
        lmps, nth, nah = hamiltonian_mc(i, j, lmps, nth, nah, dt)
    return lmps, ntp, nap, ntv, nav, nth, nah

# ---------------------
# monte carlo procedure
# ---------------------


def gen_sample(i, j, state):
    ''' generates a monte carlo sample '''
    # initialize lammps object
    x, v, box = state[:3]
    ntp, nap, ntv, nav, nth, nah = state[3:9]
    dx, dl, dt = state[9:12]
    lmps = init_lammps(i, j, x, v, box)
    # loop through monte carlo moves
    for _ in xrange(MOD):
        lmps, ntp, nap, ntv, nav, nth, nah = move_mc(i, j, lmps, ntp, nap, ntv, nav, nth, nah,
                                                     dx, dl, dt)
    # extract system properties
    natoms, x, v, temp, pe, ke, virial, box, vol = lammps_extract(lmps)
    # close lammps and remove input file
    lmps.close()
    # acceptation ratios
    with np.errstate(invalid='ignore'):
        ap = np.nan_to_num(np.float64(nap)/np.float64(ntp))
        av = np.nan_to_num(np.float64(nav)/np.float64(ntv))
        ah = np.nan_to_num(np.float64(nah)/np.float64(nth))
    # return lammps object, tries/acceptation counts, and mc params
    return natoms, x, v, temp, pe, ke, virial, box, vol, ntp, nap, ntv, nav, nth, nah, ap, av, ah


def gen_samples():
    ''' generates all monte carlo samples '''
    if PARALLEL:
        # list of delayed operations
        operations = [delayed(gen_sample)(i, j, sys_state(i, j)) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        # submit futures to client
        futures = CLIENT.compute(operations)
        # progress bar
        if VERBOSE:
            print('performing monte carlo')
            progress(futures)
            print('\n')
        # gather results from workers
        results = CLIENT.gather(futures)
    else:
        # loop through pressures
        if VERBOSE:
            print('performing monte carlo')
        results = [gen_sample(i, j, sys_state(i, j)) for i in xrange(NPRESS) for j in xrange(NTEMP)]
    return results


def update_samples(results):
    ''' updates all samples '''
    k = 0
    for i in xrange(NPRESS):
        for j in xrange(NTEMP):
            NATOMS[i, j], X[i, j], V[i, j] = results[k][:3]
            TEMP[i, j], PE[i, j], KE[i, j], VIRIAL[i, j], BOX[i, j], VOL[i, j] = results[k][3:9]
            NTP[i, j], NAP[i, j] = results[k][9:11]
            NTV[i, j], NAV[i, j] = results[k][11:13]
            NTH[i, j], NAH[i, j] = results[k][13:15]
            AP[i, j], AV[i, j], AH[i, j] = results[k][15:18]
            k += 1

# ----------------------------
# monte carlo parameter update
# ----------------------------


def gen_mc_param(state):
    ''' generate adaptive monte carlo parameters for a sample '''
    # update position displacment for pos-mc
    ap, av, ah, dx, dl, dt = state
    if ap < 0.5:
        dx = 0.9375*dx
    else:
        dx = 1.0625*dx
    # update box displacement for vol-mc
    if av < 0.5:
        dl = 0.9375*dl
    else:
        dl = 1.0625*dl
    # update timestep for hmc
    if ah < 0.5:
        dt = 0.9375*dt
    else:
        dt = 1.0625*dt
    return dx, dl, dt


def gen_mc_params():
    ''' generate adaptive monte carlo parameters for all samples '''
    if PARALLEL:
        # list of delayed operations
        operations = [delayed(gen_mc_param)(mc_state(i, j)) for i in xrange(NPRESS) for j in xrange(NTEMP)]
        # submit futures to client
        futures = CLIENT.compute(operations)
        # progress bar
        if VERBOSE:
            print('updating mc params')
            progress(futures)
            print('\n')
        # gather results from workers
        results = CLIENT.gather(futures)
    else:
        # loop through pressures
        if VERBOSE:
            print('updating mc params')
        results = [gen_mc_param(mc_state(i, j)) for i in xrange(NPRESS) for j in xrange(NTEMP)]
    return results


def update_mc_params(results):
    ''' update monte carlo parameters '''
    k = 0
    for i in xrange(NPRESS):
        for j in xrange(NTEMP):
            DX[i, j], DL[i, j], DT[i, j] = results[k]
            k += 1

# -----------------------------------------
# replica exchange markov chain monte carlo
# -----------------------------------------


def replica_exchange():
    ''' performs parallel tempering acrros all samples
        accepts/rejects based on enthalpy metropolis criterion '''
    # flatten system properties and constants
    natoms = NATOMS.reshape(-1)
    x = X.reshape(-1)
    v = V.reshape(-1)
    temp = TEMP.reshape(-1)
    pe = PE.reshape(-1)
    ke = KE.reshape(-1)
    virial = VIRIAL.reshape(-1)
    box = BOX.reshape(-1)
    vol = VOL.reshape(-1)
    et = ET.reshape(-1)
    pf = PF.reshape(-1)
    # catalog swaps
    swaps = 0
    # loop through upper right triangular matrix
    for i in xrange(len(pf)):
        for j in xrange(i+1, len(pf)):
            # configuration energies
            etoti = pe[i]+ke[i]
            etotj = pe[j]+ke[j]
            # change in enthalpy
            dH = (etoti-etotj)*(1/et[i]-1/et[j])+(pf[i]-pf[j])*(vol[i]-vol[j])
            if np.random.rand() <= np.min([1, np.exp(dH)]):
                swaps += 1
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
    if VERBOSE:
        print('%d replica exchanges performed' % swaps)
    # reshape system properties and constants
    NATOMS[:, :] = natoms.reshape((NPRESS, NTEMP))
    X[:, :] = x.reshape((NPRESS, NTEMP))
    V[:, :] = v.reshape((NPRESS, NTEMP))
    TEMP[:, :] = temp.reshape((NPRESS, NTEMP))
    PE[:, :] = pe.reshape((NPRESS, NTEMP))
    KE[:, :] = ke.reshape((NPRESS, NTEMP))
    VIRIAL[:, :] = virial.reshape((NPRESS, NTEMP))
    BOX[:, :] = box.reshape((NPRESS, NTEMP))
    VOL[:, :] = vol.reshape((NPRESS, NTEMP))

# -----------
# monte carlo
# -----------


def init_simulation():
    ''' full initialization of simulation '''
    set_constants(init_constants())
    set_outputs(init_outputs())
    init_headers()
    set_samples(init_samples())


def step_simulation():
    ''' full monte carlo step '''
    if VERBOSE:
        print('step: %d' % i)
    # collect samples for all configurations
    update_samples(gen_samples())
    # update monte carlo parameters
    update_mc_params(gen_mc_params())
    # write to output files
    write_outputs()
    # perform replica exchange markov chain monte carlo (parallel tempering)
    replica_exchange()

# initialize simulation
init_simulation()
# loop through to number of samples that need to be collected
for i in xrange(NSMPL):
    step_simulation()
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
                    if k > (NATOMS[i, j]+1)*CUTOFF:
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
