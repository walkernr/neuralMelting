# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 04:20:00 2018

@author: Nicholas
"""

import argparse
import os
import warnings
import time
import numpy as np
import numba as nb
from tqdm import tqdm
from lammps import lammps

# --------------
# run parameters
# --------------


def parse_args():
    ''' parse command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output mode', action='store_true')
    parser.add_argument('-r', '--restart', help='restart run mode', action='store_true')
    parser.add_argument('-p', '--parallel', help='parallel run mode', action='store_true')
    parser.add_argument('-c', '--client', help='dask client run mode', action='store_true')
    parser.add_argument('-d', '--distributed', help='distributed run mode', action='store_true')
    parser.add_argument('-is', '--interpolate_states', help='interpolate initial states', action='store_true')
    parser.add_argument('-bm', '--bulk_move', help='bulk position monte carlo moves', action='store_true')
    parser.add_argument('-rd', '--restart_dump', help='restart dump frequency',
                        type=int, default=128)
    parser.add_argument('-rn', '--restart_name', help='restart dump simulation name',
                        type=str, default='remcmc_init')
    parser.add_argument('-rs', '--restart_step', help='restart dump start step',
                        type=int, default=1024)
    parser.add_argument('-q', '--queue', help='job submission queue',
                        type=str, default='jobqueue')
    parser.add_argument('-a', '--allocation', help='job submission allocation',
                        type=str, default='startup')
    parser.add_argument('-nn', '--nodes', help='job node count',
                        type=int, default=1)
    parser.add_argument('-np', '--procs_per_node', help='number of processors per node',
                        type=int, default=20)
    parser.add_argument('-w', '--walltime', help='job walltime',
                        type=int, default=72)
    parser.add_argument('-m', '--memory', help='job memory (total)',
                        type=int, default=32)
    parser.add_argument('-nw', '--workers', help='job worker count (total)',
                        type=int, default=20)
    parser.add_argument('-nt', '--threads', help='threads per worker',
                        type=int, default=1)
    parser.add_argument('-mt', '--method', help='parallelization method',
                        type=str, default='fork')
    parser.add_argument('-n', '--name', help='simulation name',
                        type=str, default='remcmc_init')
    parser.add_argument('-e', '--element', help='simulation element',
                        type=str, default='LJ')
    parser.add_argument('-ss', '--supercell_size', help='simulation supercell size',
                        type=int, default=5)
    parser.add_argument('-pn', '--pressure_number', help='number of pressures',
                        type=int, default=16)
    parser.add_argument('-pr', '--pressure_range', help='pressure range (low and high)',
                        type=float, nargs=2, default=[1, 8])
    parser.add_argument('-tn', '--temperature_number', help='number of temperatures',
                        type=int, default=16)
    parser.add_argument('-tr', '--temperature_range', help='temperature range (low and high)',
                        type=float, nargs=2, default=[0.25, 2.5])
    parser.add_argument('-sc', '--sample_cutoff', help='sample recording cutoff',
                        type=int, default=0)
    parser.add_argument('-sn', '--sample_number', help='number of samples to generate',
                        type=int, default=1024)
    parser.add_argument('-sm', '--sample_mod', help='sample collection frequency',
                        type=int, default=128)
    parser.add_argument('-pm', '--position_move', help='position monte carlo move probability',
                        type=float, default=0.125)
    parser.add_argument('-vm', '--volume_move', help='volume monte carlo move probability',
                        type=float, default=0.125)
    parser.add_argument('-ts', '--timesteps', help='hamiltonian monte carlo timesteps',
                        type=int, default=8)
    parser.add_argument('-dx', '--pos_displace', help='position displacement (lattice proportion)',
                        type=float, default=0.125)
    parser.add_argument('-dv', '--vol_displace', help='logarithmic volume displacement (logarithmic volume proportion)',
                        type=float, default=0.03125)
    # parse arguments
    args = parser.parse_args()
    # return arguments
    return (args.verbose, args.restart, args.parallel, args.client, args.distributed,
            args.interpolate_states, args.bulk_move,
            args.restart_dump, args.restart_name, args.restart_step,
            args.queue, args.allocation, args.nodes, args.procs_per_node,
            args.walltime, args.memory,
            args.workers, args.threads, args.method,
            args.name, args.element, args.supercell_size,
            args.pressure_number, *args.pressure_range,
            args.temperature_number, *args.temperature_range,
            args.sample_cutoff, args.sample_number, args.sample_mod,
            args.position_move, args.volume_move, args.timesteps,
            args.pos_displace, args.vol_displace)


def client_info():
    ''' print client info '''
    info = str(CLIENT.scheduler_info)
    info = info.replace('<', '').replace('>', '').split()[6:8]
    print('\n%s %s' % tuple(info))

# ----------------
# unit definitions
# ----------------


def init_constant(k):
    ''' calculates thermodynamic constants for a sample '''
    # extract pressure/temperature indices from index
    i, j = np.unravel_index(k, dims=(NP, NT), order='C')
    if UNITS[EL] == 'real':
        na = 6.0221409e23                              # avagadro number [num/mol]
        kb = 3.29983e-27                               # boltzmann constant [kcal/K]
        r = kb*na                                      # gas constant [kcal/(mol K)]
        et = r*T[j]                                    # thermal energy [kcal/mol]
        pf = 1e-30*(1.01325e5*P[i])/(4.184e3*kb*T[j])  # metropolis prefactor [1/A^3]
    if UNITS[EL] == 'metal':
        kb = 8.61733e-5                                # boltzmann constant [eV/K]
        et = kb*T[j]                                   # thermal energy [eV]
        pf = 1e-30*(1e5*P[i])/(1.60218e-19*kb*T[j])    # metropolis prefactor [1/A^3]
    if UNITS[EL] == 'lj':
        kb = 1.0                                       # boltzmann constant (unitless)
        et = kb*T[j]                                   # thermal energy [T*]
        pf = P[i]/(kb*T[j])                            # metropolis prefactor [1/r*^3]
    return et, pf


def init_constants():
    ''' calculates thermodynamic constants for all samples '''
    if VERBOSE:
        print('----------------------')
        print('initializing constants')
        print('----------------------')
    return [init_constant(k) for k in range(NS)]

# -----------------------------
# output file utility functions
# -----------------------------


def file_prefix(i, j):
    ''' returns filename prefix for simulation '''
    prefix = os.getcwd()+'/%s.%s.%s.%02d.%02d.lammps' % (NAME, EL.lower(), LAT[EL][0], i, j)
    return prefix


def init_output(k):
    ''' initializes output filenames for a sample '''
    # extract pressure/temperature indices from index
    i, j = np.unravel_index(k, dims=(NP, NT), order='C')
    thrm = file_prefix(i, j)+'.thrm'
    traj = thrm.replace('thrm', 'traj')
    # clean old output files if they exist
    if os.path.isfile(thrm):
        os.remove(thrm)
    if os.path.isfile(thrm):
        os.remove(traj)
    return thrm, traj


def init_outputs():
    ''' initializes output filenames for all samples '''
    if VERBOSE:
        print('initializing outputs')
        print('--------------------')
    return [init_output(k) for k in range(NS)]


def init_header(k, output):
    ''' writes header for a sample '''
    # extract pressure/temperature indices from index
    i, j = np.unravel_index(k, dims=(NP, NT), order='C')
    with open(output[0], 'w') as thrm_out:
        thrm_out.write('# ---------------------\n')
        thrm_out.write('# simulation parameters\n')
        thrm_out.write('# ---------------------\n')
        thrm_out.write('# nsmpl:    %d\n' % NSMPL)
        thrm_out.write('# cutoff:   %d\n' % CUTOFF)
        thrm_out.write('# mod:      %d\n' % MOD)
        thrm_out.write('# nswps:    %d\n' % NSWPS)
        thrm_out.write('# ppos:     %f\n' % PPOS)
        thrm_out.write('# pvol:     %f\n' % PVOL)
        thrm_out.write('# phmc:     %f\n' % PHMC)
        thrm_out.write('# nstps:    %d\n' % NSTPS)
        thrm_out.write('# seed:     %d\n' % SEED)
        thrm_out.write('# ---------------------\n')
        thrm_out.write('# material properties\n')
        thrm_out.write('# ---------------------\n')
        thrm_out.write('# element:  %s\n' % EL)
        thrm_out.write('# units:    %s\n' % UNITS[EL])
        thrm_out.write('# lattice:  %s\n' % LAT[EL][0])
        thrm_out.write('# latpar:   %f\n' % LAT[EL][1])
        thrm_out.write('# size:     %d\n' % SZ)
        thrm_out.write('# mass:     %f\n' % MASS[EL])
        thrm_out.write('# press:    %f\n' % P[i])
        thrm_out.write('# temp:     %f\n' % T[j])
        thrm_out.write('# dx:       %f\n' % DX)
        thrm_out.write('# dv:       %f\n' % DV)
        thrm_out.write('# dt:       %f\n' % DT)
        thrm_out.write('# -----------------------------------------------------------------------------------------------\n')
        thrm_out.write('# | tmp | pe | ke | vir | vol | dx | dv | dt | ntp | nap | ntv | nav | nth | nah | ap | av | ah |\n')
        thrm_out.write('# -----------------------------------------------------------------------------------------------\n')


def init_headers():
    ''' writes headers for all samples '''
    if DASK:
        operations = [delayed(init_header)(k, OUTPUT[k]) for k in range(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('initializing headers')
            print('--------------------')
            progress(futures)
    elif PARALLEL:
        operations = [delayed(init_header)(k, OUTPUT[k]) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        if VERBOSE:
            print('initializing headers')
            print('--------------------')
            for k in tqdm(range(NS)):
                init_header(k, OUTPUT[k])
        else:
            for k in range(NS):
                init_header(k, OUTPUT[k])


def write_thrm(output, state):
    ''' writes thermodynamic properties to thrm file '''
    thrm = output[0]
    temp, pe, ke, virial = state[3:7]
    vol = state[8]
    dx, dv, dt = state[9:12]
    ntp, nap, ntv, nav, nth, nah = state[12:18]
    ap, av, ah = state[18:21]
    args = temp, pe, ke, virial, vol, dx, dv, dt, ntp, nap, ntv, nav, nth, nah, ap, av, ah
    with open(thrm, 'a') as thrm_out:
        thrm_out.write(len(args)*' %.4E' % args +'\n')


def write_traj(output, state):
    ''' writes trajectory data to traj file '''
    traj = output[1]
    natoms, x = state[:2]
    box = state[7]
    with open(traj, 'a') as traj_out:
        traj_out.write('%d %.4E\n' % (natoms, box))
        for i in range(natoms):
            traj_out.write(3*' %.4E' % tuple(x[3*i:3*i+3])+'\n')


def write_output(output, state):
    ''' writes output for a sample '''
    write_thrm(output, state)
    write_traj(output, state)


def write_outputs():
    ''' writes outputs for all samples '''
    if DASK:
        operations = [delayed(write_output)(OUTPUT[k], STATE[k]) for k in range(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('\n---------------')
            print('writing outputs')
            print('---------------')
            progress(futures)
    elif PARALLEL:
        operations = [delayed(write_output)(OUTPUT[k], STATE[k]) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        if VERBOSE:
            print('writing outputs')
            print('---------------')
            for k in tqdm(range(NS)):
                write_output(OUTPUT[k], STATE[k])
        else:
            for k in range(NS):
                write_output(OUTPUT[k], STATE[k])


def consolidate_outputs():
    ''' consolidates outputs across samples '''
    if VERBOSE:
        print('---------------------')
        print('consolidating outputs')
        print('---------------------')
    thrm = [OUTPUT[k][0] for k in range(NS)]
    traj = [OUTPUT[k][1] for k in range(NS)]
    with open(PREF+'.thrm', 'w') as thrm_out:
        for i in range(NP):
            for j in range(NT):
                k = np.ravel_multi_index((i, j), (NP, NT), order='C')
                with open(thrm[k], 'r') as thrm_in:
                    for line in thrm_in:
                        thrm_out.write(line)
    with open(PREF+'.traj', 'w') as traj_out:
        for i in range(NP):
            for j in range(NT):
                k = np.ravel_multi_index((i, j), (NP, NT), order='C')
                with open(traj[k], 'r') as traj_in:
                    for line in traj_in:
                        traj_out.write(line)
    if VERBOSE:
        print('cleaning files')
        print('--------------')
    for k in range(NS):
        os.remove(thrm[k])
        os.remove(traj[k])

# ------------------------------------------------
# sample initialization and information extraction
# ------------------------------------------------


def lammps_input():
    ''' constructs input file for lammps '''
    lj_param = (1.0, 1.0)
    # convert lattice definition list to strings
    prefix = os.getcwd()+'/%s.%s.%s.lammps' % (NAME, EL.lower(), LAT[EL][0])
    # set lammps file name
    lmpsfile = prefix+'.in'
    # open lammps file
    with open(lmpsfile, 'w') as lf:
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
        pot_dir = '../potentials/'
        pot_inp = (pc_pref, pot_dir, pot_dir, EL)
        if EL == 'Ti':
            lf.write('pair_style meam/c\n')
            lf.write('mass 1 47.867\n')
            lf.write('%s %slibrary.meam Ti Al %sTiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % pot_inp)
        if EL == 'Al':
            lf.write('pair_style meam/c\n')
            lf.write('mass 1 %f\n' % MASS[EL])
            lf.write('%s %slibrary.meam Ti Al %sTiAl_Kim_Kim_Jung_Lee_2016.meam %s\n\n' % pot_inp)
        if EL == 'Ni':
            lf.write('pair_style meam/c\n')
            lf.write('mass 1 %f\n' % MASS[EL])
            lf.write('%s %slibrary.Ni.meam Ni %sNi.meam %s\n\n' % pot_inp)
        if EL == 'Cu':
            lf.write('pair_style meam/c\n')
            lf.write('mass 1 %f\n' % MASS[EL])
            lf.write('%s %slibrary.Cu.meam Cu %sCu.meam %s\n\n' % pot_inp)
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
    x = np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3))
    v = np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3))
    temp = lmps.extract_compute('thermo_temp', 0, 0)
    pe = lmps.extract_compute('thermo_pe', 0, 0)
    ke = lmps.extract_compute('thermo_ke', 0, 0)
    virial = lmps.extract_compute('thermo_press', 0, 0)
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    vol = np.power(box, 3)
    return natoms, x, v, temp, pe, ke, virial, box, vol


def init_sample(k):
    ''' initializes sample '''
    i, j = np.unravel_index(k, dims=(NP, NT), order='C')
    seed = np.random.randint(1, 2**16)
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(LMPSF)
    # minimize lattice structure
    lmps.command('unfix 1')
    lmps.command('fix 1 all box/relax iso %f vmax %f' % (P[i], 0.0009765625))
    lmps.command('minimize 0.0 %f %d %d' % (1.49011612e-8, 1024, 8192))
    lmps.command('run 0')
    # randomize positions
    lmps.command('displace_atoms all random %f %f %f %d units box' % (3*(DX*LAT[EL][1],)+(seed,)))
    lmps.command('run 0')
    if INTSTS:
        # resize box
        natoms, x, v, temp, pe, ke, virial, box, vol = lammps_extract(lmps)
        volnew = np.exp(np.log(vol)+0.75*(j+1)/NT)
        boxnew = np.cbrt(volnew)
        scale = boxnew/box
        xnew = scale*x
        box_cmd = 'change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box'
        lmps.command(box_cmd % (3*(boxnew,)))
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
        lmps.command('run 0')
        # run dynamics
        lmps.command('velocity all create %f %d dist gaussian' % (T[j], seed))
        lmps.command('velocity all zero linear')
        lmps.command('velocity all zero angular')
        lmps.command('timestep %f' % DT)
        lmps.command('run 1024')
    # extract all system info
    natoms, x, v, temp, pe, ke, virial, box, vol = lammps_extract(lmps)
    lmps.close()
    ntp, nap, ntv, nav, nth, nah, ap, av, ah = np.zeros(9)
    dx, dv, dt = DX, DV, DT
    # return system info and data storage files
    return [natoms, x, v, temp, pe, ke, virial, box, vol, dx, dv, dt,
            ntp, nap, ntv, nav, nth, nah, ap, av, ah]


def init_samples():
    ''' initializes all samples '''
    if DASK:
        operations = [delayed(init_sample)(k) for k in range(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('initializing samples')
            print('--------------------')
            progress(futures)
            print('\n')
    elif PARALLEL:
        operations = [delayed(init_sample)(k) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        if VERBOSE:
            print('initializing samples')
            print('--------------------')
            futures = [init_sample(k) for k in tqdm(range(NS))]
        else:
            futures = [init_sample(k) for k in range(NS)]
    return futures


def init_lammps(x, v, box):
    ''' initializes lammps '''
    # initialize lammps
    lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmps.file(LMPSF)
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
    x = np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3))
    pe = lmps.extract_compute('thermo_pe', 0, 0)/et
    seed = np.random.randint(1, 2**16)
    lmps.command('displace_atoms all random %f %f %f %d units box' % (3*(dx*LAT[EL][1],)+(seed,)))
    lmps.command('run 0')
    penew = lmps.extract_compute('thermo_pe', 0, 0)/et
    de = penew-pe
    with np.errstate(over='ignore'):
        metcrit = np.exp(-de)
        if not np.isinf(metcrit):
            if np.random.rand() <= np.min([1, np.exp(-de)]):
                # update pos acceptations
                nap += 1
            else:
                # revert physical properties
                lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
                lmps.command('run 0')
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
    # get box dimensions
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    # get positions
    x = np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3))
    # loop through atoms
    for k in range(natoms):
        # update position tries
        ntp += 1
        # save potential energy
        pe = lmps.extract_compute('thermo_pe', 0, 0)/et
        # generate proposed positions
        od = x[3*k:3*k+3]
        nd = od+2*(np.random.rand(3)-0.5)*dx*LAT[EL][1]
        nd -= np.floor(nd/box)*box
        x[3*k:3*k+3] = nd
        lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
        lmps.command('run 0')
        # save new physical properties
        penew = lmps.extract_compute('thermo_pe', 0, 0)/et
        # calculate energy criterion
        de = penew-pe
        with np.errstate(over='ignore'):
            metcrit = np.exp(-de)
            if not np.isinf(metcrit):
                if np.random.rand() <= np.min([1, np.exp(-de)]):
                    # update pos acceptations
                    nap += 1
                else:
                    # revert physical properties
                    x[3*k:3*k+3] = od
                    lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
                    lmps.command('run 0')
            else:
                # revert physical properties
                x[3*k:3*k+3] = od
                lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
                lmps.command('run 0')
    # return lammps object and tries/acceptations
    return lmps, ntp, nap


def volume_mc(lmps, et, pf, ntv, nav, dv):
    ''' isobaric-isothermal volume monte carlo '''
    # update volume tries
    ntv += 1
    # save current physical properties
    natoms = lmps.extract_global('natoms', 0)
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    vol = np.power(box, 3)
    x = np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3))
    pe = lmps.extract_compute('thermo_pe', 0, 0)/et
    # save new physical properties
    volnew = np.exp(np.log(vol)+2*(np.random.rand()-0.5)*dv)
    boxnew = np.cbrt(volnew)
    scale = boxnew/box
    xnew = scale*x
    # apply new physical properties
    box_cmd = 'change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box'
    lmps.command(box_cmd % (3*(boxnew,)))
    lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
    lmps.command('run 0')
    penew = lmps.extract_compute('thermo_pe', 0, 0)/et
    # calculate enthalpy criterion
    dh = (penew-pe)+pf*(volnew-vol)-(natoms+1)*np.log(volnew/vol)
    # dh = (penew-pe)+pf*(volnew-vol)-natoms*np.log(volnew/vol)
    with np.errstate(over='ignore'):
        metcrit = np.exp(-dh)
        if not np.isinf(metcrit):
            if np.random.rand() <= np.min([1, np.exp(-dh)]):
                # update volume acceptations
                nav += 1
            else:
                # revert physical properties
                lmps.command(box_cmd % (3*(box,)))
                lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
                lmps.command('run 0')
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
    x = np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3))
    v = np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3))
    pe = lmps.extract_compute('thermo_pe', 0, 0)/et
    ke = lmps.extract_compute('thermo_ke', 0, 0)/et
    etot = pe+ke
    # run md
    lmps.command('run %d' % NSTPS)  # this part should be implemented as parallel
    # set new physical properties
    penew = lmps.extract_compute('thermo_pe', 0, 0)/et
    kenew = lmps.extract_compute('thermo_ke', 0, 0)/et
    etotnew = penew+kenew
    # calculate hamiltonian criterion
    de = etotnew-etot
    with np.errstate(over='ignore'):
        metcrit = np.exp(-de)
        if not np.isinf(metcrit):
            if np.random.rand() <= np.min([1, metcrit]):
                # update hamiltonian acceptations
                nah += 1
            else:
                # revert physical properties
                lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
                lmps.scatter_atoms('v', 1, 3, np.ctypeslib.as_ctypes(v))
                lmps.command('run 0')
        else:
            # revert physical properties
            lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
            lmps.scatter_atoms('v', 1, 3, np.ctypeslib.as_ctypes(v))
            lmps.command('run 0')
    # return lammps object and tries/acceptations
    return lmps, nth, nah


def move_mc(lmps, et, pf, t, ntp, nap, ntv, nav, nth, nah, dx, dv, dt):
    ''' performs monte carlo moves '''
    roll = np.random.rand()
    # position monte carlo
    if roll <= PPOS:
        if BM:
            lmps, ntp, nap = bulk_position_mc(lmps, et, ntp, nap, dx)
        else:
            lmps, ntp, nap = iter_position_mc(lmps, et, ntp, nap, dx)
    # volume monte carlo
    elif roll <= (PPOS+PVOL):
        lmps, ntv, nav = volume_mc(lmps, et, pf, ntv, nav, dv)
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
    _, j = np.unravel_index(k, dims=(NP, NT), order='C')
    t = T[j]
    et, pf = const
    x, v = state[1:3]
    box = state[7]
    dx, dv, dt = state[9:12]
    ntp, nap, ntv, nav, nth, nah = state[12:18]
    lmps = init_lammps(x, v, box)
    # loop through monte carlo moves
    for _ in range(MOD):
        lmps, ntp, nap, ntv, nav, nth, nah = move_mc(lmps, et, pf, t,
                                                     ntp, nap, ntv, nav, nth, nah, dx, dv, dt)
    # extract system properties
    natoms, x, v, temp, pe, ke, virial, box, vol = lammps_extract(lmps)
    # close lammps and remove input file
    lmps.close()
    # acceptation ratios
    with np.errstate(invalid='ignore'):
        ap = np.nan_to_num(np.float32(nap)/np.float32(ntp))
        av = np.nan_to_num(np.float32(nav)/np.float32(ntv))
        ah = np.nan_to_num(np.float32(nah)/np.float32(nth))
    # return lammps object, tries/acceptation counts, and mc params
    return [natoms, x, v, temp, pe, ke, virial, box, vol, dx, dv, dt,
            ntp, nap, ntv, nav, nth, nah, ap, av, ah]


def gen_samples():
    ''' generates all monte carlo samples '''
    if DASK:
        # list of delayed operations
        operations = [delayed(gen_sample)(k, CONST[k], STATE[k]) for k in range(NS)]
        # submit futures to client
        futures = CLIENT.compute(operations)
        # progress bar
        if VERBOSE:
            print('----------------------')
            print('performing monte carlo')
            print('----------------------')
            progress(futures)
    elif PARALLEL:
        operations = [delayed(gen_sample)(k, CONST[k], STATE[k]) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        # loop through pressures
        if VERBOSE:
            print('----------------------')
            print('performing monte carlo')
            print('----------------------')
            futures = [gen_sample(k, CONST[k], STATE[k]) for k in tqdm(range(NS))]
        else:
            futures = [gen_sample(k, CONST[k], STATE[k]) for k in range(NS)]
    return futures

# ----------------------------
# monte carlo parameter update
# ----------------------------


def gen_mc_param(state):
    ''' generate adaptive monte carlo parameters for a sample '''
    # update position displacment for pos-mc
    dx, dv, dt = state[9:12]
    ap, av, ah = state[-3:]
    if ap < 0.5:
        dx = 0.9375*dx
    if ap > 0.5:
        dx = 1.0625*dx
    # update box displacement for vol-mc
    if av < 0.5:
        dv = 0.9375*dv
    if av > 0.5:
        dv = 1.0625*dv
    # update timestep for hmc
    if ah < 0.5:
        dt = 0.9375*dt
    if ah > 0.5:
        dt = 1.0625*dt
    return state[:9]+[dx, dv, dt]+list(np.zeros(9))


def gen_mc_params():
    ''' generate adaptive monte carlo parameters for all samples '''
    if DASK:
        # list of delayed operations
        operations = [delayed(gen_mc_param)(STATE[k]) for k in range(NS)]
        # submit futures to client
        futures = CLIENT.compute(operations)
        # progress bar
        if VERBOSE:
            print('\n------------------')
            print('updating mc params')
            print('------------------')
            progress(futures)
    elif PARALLEL:
        operations = [delayed(gen_mc_param)(STATE[k]) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        # loop through pressures
        if VERBOSE:
            print('updating mc params')
            print('------------------')
        futures = [gen_mc_param(STATE[k]) for k in range(NS)]
    return futures

# -----------------------------------------
# replica exchange markov chain monte carlo
# -----------------------------------------

@nb.jit
def replica_exchange():
    ''' performs parallel tempering across temperature samples for each pressure '''
    # catalog swaps
    swaps = 0
    # loop through pressures
    for u in range(NP):
        # loop through reference temperatures from high to low
        for v in range(NT-1, -1, -1):
            # loop through temperatures from low to current reference temperature
            for w in range(v):
                # extract index from each pressure/temperature index pair
                i = np.ravel_multi_index((u, v), (NP, NT), order='C')
                j = np.ravel_multi_index((u, w), (NP, NT), order='C')
                # calculate energy and volume differences
                de, dv = sum(STATE[i][4:6])-sum(STATE[j][4:6]), STATE[i][8]-STATE[j][8]
                # enthalpy difference
                dh = de*(1./CONST[i][0]-1./CONST[j][0])+(CONST[i][1]-CONST[j][1])*dv
                # metropolis criterion
                if np.random.rand() <= np.min([1, np.exp(dh)]):
                    swaps += 1
                    # swap states
                    STATE[j][:12], STATE[i][:12] = STATE[i][:12], STATE[j][:12]
    if VERBOSE:
        if PARALLEL:
            print('\n-------------------------------')
        print('%d replica exchanges performed' % swaps)
        print('-------------------------------')

# -------------
# restart files
# -------------


def load_samples_restart():
    ''' initialize samples with restart file '''
    if VERBOSE:
        if PARALLEL:
            print('\n----------------------------------')
        print('loading samples from previous dump')
        print('----------------------------------')
    rf = os.getcwd()+'/%s.%s.%s.lammps.rstrt.%04d.npy' % (RENAME, EL.lower(), LAT[EL][0], RESTEP)
    return list(np.load(rf))


def dump_samples_restart():
    ''' save restart state '''
    if VERBOSE:
        if PARALLEL:
            print('\n---------------')
        print('dumping samples')
    rf = os.getcwd()+'/%s.%s.%s.lammps.rstrt.%04d.npy' % (NAME, EL.lower(), LAT[EL][0], STEP+1)
    np.save(rf, np.array(STATE, dtype=object))

# ----
# main
# ----

if __name__ == '__main__':

    (VERBOSE, RESTART,
     PARALLEL, DASK, DISTRIBUTED,
     INTSTS, BM,
     REFREQ, RENAME, RESTEP,
     QUEUE, ALLOC, NODES, PPN,
     WALLTIME, MEM,
     NWORKER, NTHREAD, MTHD,
     NAME, EL, SZ,
     NP, LP, HP,
     NT, LT, HT,
     CUTOFF, NSMPL, MOD,
     PPOS, PVOL, NSTPS,
     DX, DV) = parse_args()

    # set random seed
    SEED = 256
    np.random.seed(SEED)
    # processing or threading
    PROC = (NWORKER != 1)
    # ensure all flags are consistent
    if DISTRIBUTED and not DASK:
        DASK = 1
    if DASK and not PARALLEL:
        PARALLEL = 1

    # number of simulations
    NS = NP*NT
    # total number of monte carlo sweeps
    NSWPS = NSMPL*MOD
    # hamiltonian monte carlo probability
    PHMC = 1-PPOS-PVOL

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
    # inital time step
    DT = TIMESTEP[UNITS[EL]]
    # prefix
    PREF = os.getcwd()+'/%s.%s.%s.lammps' % (NAME, EL.lower(), LAT[EL][0])
    # dump params
    np.save(PREF+'.virial.trgt.npy', P)
    np.save(PREF+'.temp.trgt.npy', T)

    # -----------------
    # initialize lammps
    # -----------------

    LMPSF = lammps_input()

    # -----------------
    # initialize client
    # -----------------

    if PARALLEL:
        from multiprocessing import freeze_support
    if not DASK:
        from joblib import Parallel, delayed
    if DASK:
        os.environ['DASK_ALLOWED_FAILURES'] = '64'
        os.environ['DASK_WORK_STEALING'] = 'True'
        os.environ['DASK_MULTIPROCESSING_METHOD'] = MTHD
        os.environ['DASK_LOG_FORMAT'] = '\r%(name)s - %(levelname)s - %(message)s'
        from distributed import Client, LocalCluster, progress
        from dask import delayed
    if DISTRIBUTED:
        from dask_jobqueue import PBSCluster

    if PARALLEL:
        freeze_support()
        if DASK and not DISTRIBUTED:
            # construct local cluster
            CLUSTER = LocalCluster(n_workers=NWORKER, threads_per_worker=NTHREAD, processes=PROC)
            # start client with local cluster
            CLIENT = Client(CLUSTER)
            # display client information
            if VERBOSE:
                client_info()
        if DASK and DISTRIBUTED:
            # construct distributed cluster
            CLUSTER = PBSCluster(queue=QUEUE, project=ALLOC,
                                 resource_spec='nodes=%d:ppn=%d' % (NODES, PPN),
                                 walltime='%d:00:00' % WALLTIME,
                                 processes=NWORKER, cores=NTHREAD*NWORKER, memory=str(MEM)+'GB',
                                 local_dir=os.getcwd())
            CLUSTER.start_workers(1)
            # start client with distributed cluster
            CLIENT = Client(CLUSTER)
            while 'processes=0 cores=0' in str(CLIENT.scheduler_info):
                time.sleep(5)
                if VERBOSE:
                    client_info()

    # -----------
    # monte carlo
    # -----------

    # define thermodynamic constants
    CONST = init_constants()
    # define output file names
    OUTPUT = init_outputs()
    if CUTOFF < NSMPL:
        init_headers()
    # initialize simulation
    if RESTART:
        STATE = load_samples_restart()
        replica_exchange()
    else:
        if DASK:
            STATE = CLIENT.gather(init_samples())
        else:
            STATE = init_samples()
    # loop through to number of samples that need to be collected
    STEP = -1
    dump_samples_restart()
    for STEP in tqdm(range(NSMPL)):
        if VERBOSE and DASK:
            client_info()
        # generate samples
        STATE[:] = gen_samples()
        # generate mc parameters
        if (STEP+1) > CUTOFF:
            # write data
            write_outputs()
        STATE[:] = gen_mc_params()
        if DASK:
            # gather results from cluster
            STATE[:] = CLIENT.gather(STATE)
        if (STEP+1) % REFREQ == 0:
            # save state for restart
            dump_samples_restart()
        # replica exchange markov chain mc
        if (STEP+1) != NSMPL:
            replica_exchange()
    if DASK:
        # terminate client after completion
        CLIENT.close()
    # consolidate output files
    if CUTOFF < NSMPL:
        consolidate_outputs()
