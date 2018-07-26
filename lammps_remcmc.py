# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 04:20:00 2018

@author: Nicholas
"""

from __future__ import division, print_function
import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from lammps import lammps

# --------------
# run parameters
# --------------

# parse command line (help option generated automatically)
PARSER = argparse.ArgumentParser()
PARSER.add_argument('-v', '--verbose', help='verbose output', action='store_true')
PARSER.add_argument('-r', '--restart', help='restart run', action='store_true')
PARSER.add_argument('-p', '--parallel', help='parallel run', action='store_true')
PARSER.add_argument('-d', '--distributed', help='distributed run', action='store_true')
PARSER.add_argument('-rd', '--restart_dump', help='restart dump frequency',
                    type=int, default=256)
PARSER.add_arguemnt('-rn', '--resart_name', help='restart dump name'
                    type=str, default='test')
PARSER.add_argument('-rs', '--restart_step', help='restart run step',
                    type=int, default=256)
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
PARSER.add_argument('-dx', '--pos_displace', help='position displacement (proportion of lattice)',
                    type=float, default=0.125)
PARSER.add_argument('-dl', '--box_displace', help='box displacement (proportion of box)',
                    type=float, default=0.03125)
# parse arguments
ARGS = PARSER.parse_args()

# booleans for run type
VERBOSE = ARGS.verbose
PARALLEL = ARGS.parallel
RESTART = ARGS.restart
DISTRIBUTED = ARGS.distributed
# restart step
RESTEP = ARGS.restart_step
# restart write frequency
REFREQ = ARGS.restart_write
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
NP = ARGS.pressure_number
LP, HP = ARGS.pressure_range
# temperature parameters
NT = ARGS.temperature_number
LT, HT = ARGS.temperature_range
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
# proportional displacements
PDX = ARGS.pos_displace
PDL = ARGS.box_displace

if PARALLEL:
    os.environ['DASK_ALLOWED_FAILURES'] = '4'
    from distributed import Client, LocalCluster, progress
    from dask import delayed
if DISTRIBUTED:
    import time
    from dask_jobqueue import PBSCluster

# number of simulations
NS = NP*NT
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
# pressure
P = np.linspace(LP, HP, NP, dtype=np.float32)
# temperature
T = np.linspace(LT, HT, NT, dtype=np.float32)
# inital position variation, volume variation, and timestep
DX = PDX*LAT[EL][1]
DL = PDL*SZ*LAT[EL][1]
DT = TIMESTEP[UNITS[EL]]

# ----------------
# unit definitions
# ----------------


def init_constant(k):
    ''' calculates thermodynamic constants for a sample '''
    i, j = np.unravel_index(k, dims=(NP, NT), order='C')
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
    if VERBOSE:
        print('initializing constants')
    return [init_constant(k) for k in xrange(NS)]

# -----------------------------
# output file utility functions
# -----------------------------


def file_prefix(i):
    ''' returns filename prefix for simulation '''
    prefix = '%s.%s.%s.%02d.lammps' % (NAME, EL.lower(), LAT[EL][0], i)
    return os.getcwd()+'/'+prefix


def init_output(k):
    ''' initializes output filenames for a sample '''
    i, j = np.unravel_index(k, dims=(NP, NT), order='C')
    thrm = file_prefix(i)+'.%02d.thrm' % j
    traj = thrm.replace('thrm', 'traj')
    if os.path.isfile(thrm):
        os.remove(thrm)
    if os.path.isfile(thrm):
        os.remove(traj)
    return thrm, traj


def init_outputs():
    ''' initializes output filenames for all samples '''
    if VERBOSE:
        print('initializing outputs')
    return [init_output(k) for k in xrange(NS)]


def init_header(k, output):
    ''' writes header for a sample '''
    i, j = np.unravel_index(k, dims=(NP, NT), order='C')
    with open(output[0], 'wb') as thrm_out:
        thrm_out.write('#----------------------\n')
        thrm_out.write('# simulation parameters\n')
        thrm_out.write('#----------------------\n')
        thrm_out.write('# nsmpl:    %d\n' % NSMPL)
        thrm_out.write('# cutoff:   %d\n' % CUTOFF)
        thrm_out.write('# mod:      %d\n' % MOD)
        thrm_out.write('# nswps:    %d\n' % NSWPS)
        thrm_out.write('# ppos:     %f\n' % PPOS)
        thrm_out.write('# pvol:     %f\n' % PVOL)
        thrm_out.write('# phmc:     %f\n' % PHMC)
        thrm_out.write('# nstps:    %d\n' % NSTPS)
        thrm_out.write('# seed:     %d\n' % SEED)
        thrm_out.write('#----------------------\n')
        thrm_out.write('# material properties\n')
        thrm_out.write('#----------------------\n')
        thrm_out.write('# element:  %s\n' % EL)
        thrm_out.write('# units:    %s\n' % UNITS[EL])
        thrm_out.write('# lattice:  %s\n' % LAT[EL][0])
        thrm_out.write('# latpar:   %f\n' % LAT[EL][1])
        thrm_out.write('# size:     %d\n' % SZ)
        thrm_out.write('# mass:     %f\n' % MASS[EL])
        thrm_out.write('# press:    %f\n' % P[i])
        thrm_out.write('# temp:     %f\n' % T[j])
        thrm_out.write('# dx:       %f\n' % DX)
        thrm_out.write('# dl:       %f\n' % DL)
        thrm_out.write('# dt:       %f\n' % DT)
        thrm_out.write('# ------------------------------------------------------------\n')
        thrm_out.write('# | temp | pe | ke | virial | vol | accpos | accvol | acchmc |\n')
        thrm_out.write('# ------------------------------------------------------------\n')


def init_headers():
    ''' writes headers for all samples '''
    if PARALLEL:
        operations = [delayed(init_header)(k, OUTPUT[k]) for k in xrange(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('initializing headers')
            progress(futures)
            print('\n')
    else:
        if VERBOSE:
            print('initializing headers')
        for k in xrange(NS):
            init_header(k, OUTPUT[k])


def write_thrm(output, state):
    ''' writes thermodynamic properties to thrm file '''
    thrm = output[0]
    temp, pe, ke, virial = state[3:7]
    vol = state[8]
    ap, av, ah = state[15:18]
    args = temp, pe, ke, virial, vol, ap, av, ah
    with open(thrm, 'ab') as thrm_out:
        thrm_out.write('%.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n' % args)


def write_traj(output, state):
    ''' writes trajectory data to traj file '''
    traj = output[1]
    natoms, x = state[:2]
    box = state[7]
    with open(traj, 'ab') as traj_out:
        traj_out.write('%d %.4E\n' % (natoms, box))
        for i in xrange(natoms):
            traj_out.write('%.4E %.4E %.4E\n' % tuple(x[3*i:3*i+3]))


def write_output(output, state):
    ''' writes output for a sample '''
    write_thrm(output, state)
    write_traj(output, state)


def write_outputs():
    ''' writes outputs for all samples '''
    if PARALLEL:
        operations = [delayed(write_output)(OUTPUT[k], STATE[k]) for k in xrange(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('writing outputs')
            progress(futures)
            print('\n')
    else:
        if VERBOSE:
            print('writing outputs')
        for k in xrange(NS):
            write_output(OUTPUT[k], STATE[k])


def consolidate_outputs():
    ''' consolidates outputs across samples '''
    if VERBOSE:
        print('consolidating outputs')
    thrm = [OUTPUT[k][0] for k in xrange(NS)]
    traj = [OUTPUT[k][1] for k in xrange(NS)]
    for i in xrange(NP):
        with open(file_prefix(i)+'.thrm', 'wb') as thrm_out:
            for j in xrange(NT):
                k = np.ravel_multi_index((i, j), (NP, NT), order='C')
                with open(thrm[k], 'rb') as thrm_in:
                    for line in thrm_in:
                        thrm_out.write(line)
    for i in xrange(NP):
        with open(file_prefix(i)+'.traj', 'wb') as traj_out:
            for j in xrange(NT):
                k = np.ravel_multi_index((i, j), (NP, NT), order='C')
                with open(traj[k], 'rb') as traj_in:
                    for line in traj_in:
                        traj_out.write(line)
    if VERBOSE:
        print('cleaning files')
    for k in xrange(NS):
        os.remove(thrm[k])
        os.remove(thrm[k])

# ---------------------------------
# lammps file/object initialization
# ---------------------------------


def lammps_input(i):
    ''' constructs input file for lammps '''
    lj_param = (1.0, 1.0)
    # convert lattice definition list to strings
    prefix = file_prefix(i)
    # set lammps file name
    lmpsfile = prefix+'.in'
    # open lammps file
    with open(lmpsfile, 'wb') as lf:
        # file header
        lf.write('# LAMMPS Monte Carlo: %s\n\n' % EL)
        # units and atom style
        lf.write('units %s\n' % UNITS[EL])
        lf.write('atom_style atomic\n')
        lf.write('atom_modify map yes\n\n')
        # construct simulation box
        lf.write('boundary p p p\n')
        lf.write('lattice %s %s\n' % tuple(LAT[EL]))
        lf.write('region box block 0 %d 0 %d 0 %d\n' % (3*(SZ,)))
        lf.write('create_box 1 box\n')
        lf.write('create_atoms 1 box\n\n')
        # potential definitions
        pc_pref = 'pair_coeff * *'
        if EL == 'Ti':
            lf.write('pair_style meam/c\n')
            lf.write('mass 1 47.867\n')
            lf.write('%s library.meam Ti Al TiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % (pc_pref, EL))
        if EL == 'Al':
            lf.write('pair_style meam/c\n')
            lf.write('mass 1 %f\n' % MASS[EL])
            lf.write('%s library.meam Ti Al TiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % (pc_pref, EL))
        if EL == 'Ni':
            lf.write('pair_style meam/c\n')
            lf.write('mass 1 %f\n' % MASS[EL])
            lf.write('%s library.Ni.meam Ni Ni.meam %s\n\n' % (pc_pref, EL))
        if EL == 'Cu':
            lf.write('pair_style meam/c\n')
            lf.write('mass 1 %f\n' % MASS[EL])
            lf.write('%s library.Cu.meam Cu Cu.meam %s\n\n' % (pc_pref, EL))
        if EL == 'LJ':
            lf.write('pair_style lj/cut 2.5\n')
            lf.write('mass 1 %f\n' % MASS[EL])
            lf.write('pair_coeff 1 1 %f %f 2.5\n\n' % lj_param)
        # compute kinetic energy
        lf.write('compute thermo_ke all ke\n\n')
        lf.write('timestep %f\n' % DT)
        lf.write('fix 1 all nve\n')
        lf.write('run 0')
    # return file name
    return lmpsfile


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


def init_sample(k):
    ''' initializes sample '''
    i, _ = np.unravel_index(k, dims=(NP, NT), order='C')
    # generate input file
    lmpsfile = lammps_input(i)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfile)
    # minimize lattice structure
    lmps.command('unfix 1')
    lmps.command('fix 1 all box/relax iso %f vmax %f' % (P[i], 0.0009765625))
    lmps.command('minimize 0.0 %f %d %d' % (1.49011612e-8, 1024, 8192))
    seed = np.random.randint(1, 2**16)
    lmps.command('displace_atoms all random %f %f %f %d units box' % (3*(DX,)+(seed,)))
    # extract all system info
    natoms, x, v, temp, pe, ke, virial, box, vol = lammps_extract(lmps)
    lmps.close()
    ntp, nap, ntv, nav, nth, nah, ap, av, ah = np.zeros(9)
    dx, dl, dt = DL, DX, DT
    # return system info and data storage files
    return (natoms, x, v, temp, pe, ke, virial, box, vol,
            ntp, nap, ntv, nav, nth, nah, ap, av, ah, dx, dl, dt)


def init_samples():
    ''' initializes all samples '''
    if PARALLEL:
        operations = [delayed(init_sample)(k) for k in xrange(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('initializing samples')
            progress(futures)
            print('\n')
    else:
        if VERBOSE:
            print('initializing samples')
        futures = [init_sample(k) for k in xrange(NS)]
    return futures


def init_lammps(i, x, v, box):
    ''' initializes lammps '''
    # generate input file
    lmpsfile = lammps_input(i)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(lmpsfile)
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


def bulk_position_mc(lmps, et, ntp, nap, dx):
    ''' classic position monte carlo (bulk) '''
    ntp += 1
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    pe = lmps.extract_compute('thermo_pe', None, 0)/et
    seed = np.random.randint(1, 2**16)
    lmps.command('displace_atoms all random %f %f %f %d units box' % (3*(dx,)+(seed,)))
    lmps.command('run 0')
    penew = lmps.extract_compute('thermo_pe', None, 0)/et
    de = penew-pe
    if np.random.rand() <= np.min([1, np.exp(-de)]):
        # update pos acceptations
        nap += 1
    else:
        # revert physical properties
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
        lmps.command('run 0')
    # return lammps object and tries/acceptations
    return lmps, ntp, nap


def iter_position_mc(lmps, et, ntp, nap, dx):
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
        pe = lmps.extract_compute('thermo_pe', None, 0)/et
        xnew = np.copy(x)
        xnew[3*k:3*k+3] += (np.random.rand(3)-0.5)*dx
        xnew[3*k:3*k+3] -= np.floor(xnew[3*k:3*k+3]/box)*box
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
        lmps.command('run 0')
        penew = lmps.extract_compute('thermo_pe', None, 0)/et
        de = penew-pe
        if np.random.rand() <= np.min([1, np.exp(-de)]):
            # update pos acceptations
            nap += 1
        else:
            # revert physical properties
            lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
            lmps.command('run 0')
    # return lammps object and tries/acceptations
    return lmps, ntp, nap


def volume_mc(lmps, et, pf, ntv, nav, dl):
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
    pe = lmps.extract_compute('thermo_pe', None, 0)/et
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
    penew = lmps.extract_compute('thermo_pe', None, 0)/et
    # calculate enthalpy criterion
    dh = (penew-pe)+pf*(volnew-vol)-natoms*np.log(volnew/vol)
    if np.random.rand() <= np.min([1, np.exp(-dh)]):
        # update volume acceptations
        nav += 1
    else:
        # revert physical properties
        lmps.command(box_cmd % (3*(box,)))
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
        lmps.command('run 0')
    # return lammps object and tries/acceptations
    return lmps, ntv, nav


def hamiltonian_mc(lmps, et, t, nth, nah, dt):
    ''' hamiltionian monte carlo '''
    # update hmc tries
    nth += 1
    # set new atom velocities and initialize
    seed = np.random.randint(1, 2**16)
    lmps.command('velocity all create %f %d dist gaussian' % (t, seed))
    lmps.command('velocity all zero linear')
    lmps.command('velocity all zero angular')
    lmps.command('timestep %f' % dt)
    lmps.command('run 0')
    # save current physical properties
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    v = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3)))
    pe = lmps.extract_compute('thermo_pe', None, 0)/et
    ke = lmps.extract_compute('thermo_ke', None, 0)/et
    etot = pe+ke
    # run md
    lmps.command('run %d' % NSTPS)  # this part should be implemented as parallel
    # set new physical properties
    penew = lmps.extract_compute('thermo_pe', None, 0)/et
    kenew = lmps.extract_compute('thermo_ke', None, 0)/et
    etotnew = penew+kenew
    # calculate hamiltonian criterion
    de = etotnew-etot
    if np.random.rand() <= np.min([1, np.exp(-de)]):
        # update hamiltonian acceptations
        nah += 1
    else:
        # revert physical properties
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
        lmps.scatter_atoms('v', 1, 3, np.ctypeslib.as_ctypes(v))
        lmps.command('run 0')
    # return lammps object and tries/acceptations
    return lmps, nth, nah


def move_mc(lmps, et, pf, t, ntp, nap, ntv, nav, nth, nah, dx, dl, dt):
    ''' performs monte carlo moves '''
    roll = np.random.rand()
    # position monte carlo
    if roll <= PPOS:
        lmps, ntp, nap = bulk_position_mc(lmps, et, ntp, nap, dx)
        # lmps, ntp, nap = iter_position_mc(lmps, et, ntp, nap, dx)
    # volume monte carlo
    elif roll <= (PPOS+PVOL):
        lmps, ntv, nav = volume_mc(lmps, et, pf, ntv, nav, dl)
    # hamiltonian monte carlo
    else:
        lmps, nth, nah = hamiltonian_mc(lmps, et, t, nth, nah, dt)
    return lmps, ntp, nap, ntv, nav, nth, nah

# ---------------------
# monte carlo procedure
# ---------------------


def gen_sample(k, const, state):
    ''' generates a monte carlo sample '''
    # initialize lammps object
    i, j = np.unravel_index(k, dims=(NP, NT), order='C')
    t = T[j]
    et, pf = const
    x, v = state[1:3]
    box = state[7]
    ntp, nap, ntv, nav, nth, nah = state[9:15]
    dx, dl, dt = state[18:21]
    lmps = init_lammps(i, x, v, box)
    # loop through monte carlo moves
    for _ in xrange(MOD):
        lmps, ntp, nap, ntv, nav, nth, nah = move_mc(lmps, et, pf, t,
                                                     ntp, nap, ntv, nav, nth, nah, dx, dl, dt)
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
    return (natoms, x, v, temp, pe, ke, virial, box, vol,
            ntp, nap, ntv, nav, nth, nah, ap, av, ah, dx, dl, dt)


def gen_samples():
    ''' generates all monte carlo samples '''
    if PARALLEL:
        # list of delayed operations
        operations = [delayed(gen_sample)(k, CONST[k], STATE[k]) for k in xrange(NS)]
        # submit futures to client
        futures = CLIENT.compute(operations)
        # progress bar
        if VERBOSE:
            print('performing monte carlo')
            progress(futures)
            print('\n')
    else:
        # loop through pressures
        if VERBOSE:
            print('performing monte carlo')
        futures = [gen_sample(k, CONST[k], STATE[k]) for k in xrange(NS)]
    return futures

# ----------------------------
# monte carlo parameter update
# ----------------------------


def gen_mc_param(state):
    ''' generate adaptive monte carlo parameters for a sample '''
    # update position displacment for pos-mc
    ap, av, ah, dx, dl, dt = state[-6:]
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
    return state[:-6]+(ap, av, ah, dx, dl, dt)


def gen_mc_params():
    ''' generate adaptive monte carlo parameters for all samples '''
    if PARALLEL:
        # list of delayed operations
        operations = [delayed(gen_mc_param)(STATE[k]) for k in xrange(NS)]
        # submit futures to client
        futures = CLIENT.compute(operations)
        # progress bar
        if VERBOSE:
            print('updating mc params')
            progress(futures)
            print('\n')
    else:
        # loop through pressures
        if VERBOSE:
            print('updating mc params')
        futures = [gen_mc_param(STATE[k]) for k in xrange(NS)]
    return futures

# -----------------------------------------
# replica exchange markov chain monte carlo
# -----------------------------------------


def replica_exchange():
    ''' performs parallel tempering acrros all samples
        accepts/rejects based on enthalpy metropolis criterion '''
    # collect system properties
    et = [CONST[k][0] for k in xrange(NS)]
    pf = [CONST[k][1] for k in xrange(NS)]
    etot = [STATE[k][4]+STATE[k][5] for k in xrange(NS)]
    vol = [STATE[k][8] for k in xrange(NS)]
    # catalog swaps
    swaps = 0
    # loop through upper right triangular matrix
    for i in xrange(NS):
        for j in xrange(i+1, NS):
            # change in enthalpy
            de, dv = etot[i]-etot[j], vol[i]-vol[j]
            dh = de*(1/et[i]-1/et[j])+(pf[i]-pf[j])*dv
            if np.random.rand() <= np.min([1, np.exp(dh)]):
                swaps += 1
                # swap lammps objects
                etot[j], etot[i] = etot[i], etot[j]
                vol[j], vol[i] = vol[i], vol[j]
                STATE[j], STATE[i] = STATE[i], STATE[j]
    if VERBOSE:
        print('%d replica exchanges performed' % swaps)

# -------------
# restart files
# -------------


def load_samples_restart():
    ''' initialize samples with restart file '''
    if VERBOSE:
        print('loading samples from previous dump')
    rf = os.getcwd()+'/'+'%s.%s.%s.lammps.rstrt.%d.pickle' % (NAME, EL.lower(), LAT[EL][0], RESTEP)
    return pickle.load(open(rf, 'rb'))


def dump_samples_restart():
    ''' save restart state '''
    if VERBOSE:
        print('dumping samples')
    rf = os.getcwd()+'/'+'%s.%s.%s.lammps.rstrt.%d.pickle' % (NAME, EL.lower(), LAT[EL][0], STEP)
    pickle.dump(STATE, open(rf, 'wb'))

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
        if NWORKER == 1:
            PROC = False
        else:
            PROC = True
        CLUSTER = LocalCluster(n_workers=NWORKER, threads_per_worker=NTHREAD, processes=PROC)
        # start client with local cluster
        CLIENT = Client(CLUSTER)
        # display client information
        if VERBOSE:
            print(CLIENT.scheduler_info)

# -----------
# monte carlo
# -----------

# define thermodynamic constants
CONST = init_constants()
# define output file names
OUTPUT = init_outputs()
init_headers()
# initialize simulation
if RESTART:
    STATE = load_samples_restart()
else:
    if PARALLEL:
        STATE = CLIENT.gather(init_samples())
    else:
        STATE = init_samples()
# loop through to number of samples that need to be collected
for STEP in tqdm(xrange(NSMPL)):
    # generate samples
    STATE = gen_samples()
    # generate mc parameters
    STATE = gen_mc_params()
    if STEP >= CUTOFF:
        # write data
        write_outputs()
    if PARALLEL:
        # gather results from cluster
        STATE = CLIENT.gather(STATE)
    if STEP % REFREQ == 0 and STEP > 0:
        # save state for restart
        dump_samples_restart()
        if PARALLEL:
            CLIENT.restart()
    # replica exchange markov chain mc
    replica_exchange()
if PARALLEL:
    # terminate client after completion
    CLIENT.close()
# consolidate output files
consolidate_outputs()
