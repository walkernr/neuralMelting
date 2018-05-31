# -*- coding: utf-8 -*-
"""
Created on Wed May 20 05:21:42 2018

@author: Nicholas
"""

from __future__ import division, print_function
import sys
import numpy as np
from lammps import lammps
#  from mpi4py import MPI

#  parallel support not currently implemented
#  comm = MPI.COMM_WORLD

# --------------
# run parameters
# --------------

# element choice
try:
    el = sys.argv[1]
except:
    el = 'LJ'
# number of data sets
n_dat = 64
# simulation name
name = 'hmc'
# monte carlo parameters
mod = 128             # frequency of data storage
n_swps = 1024*mod      # total hmc sweeps
n_stps = 16           # md steps during hmc
seed = 256            # random seed
np.random.seed(seed)  # initialize rng

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
P = {'Ti': 1.0,
     'Al': 1.0,
     'Ni': 1.0,
     'Cu': 1.0,
     'LJ': 1.0}
# temperature
T = {'Ti': np.linspace(256, 2560, n_dat, dtype=np.float64),
     'Al': np.linspace(256, 2560, n_dat, dtype=np.float64),
     'Ni': np.linspace(256, 2560, n_dat, dtype=np.float64),
     'Cu': np.linspace(256, 2560, n_dat, dtype=np.float64),
     'LJ': np.linspace(0.25, 2.5, n_dat, dtype=np.float64)}
# lattice type and parameter
lat = {'Ti': ('bcc', 2.951),
       'Al': ('fcc', 4.046),
       'Ni': ('fcc', 3.524),
       'Cu': ('fcc', 3.615),
       'LJ': ('fcc', 2**(1/6)*lj_param[1])}
# box size
sz = {'Ti': 5,
      'Al': 4,
      'Ni': 4,
      'Cu': 4,
      'LJ': 4}
# mass
mass = {'Ti': 47.867,
        'Al': 29.982,
        'Ni': 58.693,
        'Cu': 63.546,
        'LJ': 1.0}
# probability of hamiltonian monte carlo
p_hmc = {'Ti': 0.875,
         'Al': 0.875,
         'Ni': 0.875,
         'Cu': 0.875,
         'LJ': 0.875}
# max log volume change displacement
lnvol_max = {'Ti': 0.00048828125,
             'Al': 0.00048828125,
             'Ni': 0.00048828125,
             'Cu': 0.00048828125,
             'LJ': 0.00024414062}
# timestep
dt = {'real': 4.0,
      'metal': 0.00390625,
      'lj': 0.00390625}

# ---------------------------------
# lammps file construction function
# ---------------------------------

def lammps_input():
    ''' constructs input file for lammps
        takes element name, lattice definitions, size, and simulation name
        returns input file name '''
    # convert lattice definition list to strings
    prefix = '%s.%s.%d.lammps' % (el.lower(), lat[el][0], int(P[el]))
    # set lammps file name
    job = prefix+'.'+name
    lmpsfilein = job+'.in'
    # open lammps file
    lmpsfile = open(lmpsfilein, 'w')
    # file header
    lmpsfile.write('# LAMMPS Monte Carlo: %s\n\n' % el)
    # units and atom style
    lmpsfile.write('units %s\n' % units[el])
    lmpsfile.write('atom_style atomic\n')
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
        lmpsfile.write('pair_coeff * * %f %f\n\n' % lj_param)
    # minimize lattice structure
    lmpsfile.write('minimize 0.0 1.0e-8 1024 8192\n')
    # compute kinetic energy
    lmpsfile.write('compute thermo_ke all ke\n')
    # dynamics initialization
    lmpsfile.write('timestep %f\n' % dt[units[el]])
    lmpsfile.write('fix 1 all nve\n')
    lmpsfile.write('run 0\n')
    # close file and return name
    lmpsfile.close()
    return lmpsfilein

# ---------------------
# initialize lammps run
# ---------------------

# generate input file
lmpsfilein = lammps_input()
# initialize lammps
lmps = lammps(name='', cmdargs=['-log', 'none', '-screen', 'none'])
lmps.file(lmpsfilein)
# extract atom information
natoms = lmps.extract_global('natoms', 0)
atomid = np.ctypeslib.as_array(lmps.gather_atoms('id', 0, 1))
atype = np.ctypeslib.as_array(lmps.gather_atoms('type', 0, 1))
# open data storage files
thermo = open(lmpsfilein.replace('.in', '.thrm'), 'w')
traj = open(lmpsfilein.replace('.in', '.traj'), 'w')

thermo.write('#----------------------\n')
thermo.write('# simulation parameters\n')
thermo.write('#----------------------\n')
thermo.write('# ndat:     %d\n' % n_dat)
thermo.write('# mod:      %d\n' % mod)
thermo.write('# nswps:    %d\n' % n_swps)
thermo.write('# nstps:    %d\n' % n_stps)
thermo.write('# seed:     %d\n' % seed)
thermo.write('#----------------------\n')
thermo.write('# material properties\n')
thermo.write('#----------------------\n')
thermo.write('# element:  %s\n' % el)
thermo.write('# units:    %s\n' % units[el])
thermo.write('# lattice:  %s\n' % lat[el][0])
thermo.write('# latpar:   %f\n' % lat[el][1])
thermo.write('# size:     %d\n' % sz[el])
thermo.write('# mass:     %f\n' % mass[el])
thermo.write('# press:    %f\n' % P[el])
thermo.write('# mintemp:  %f\n' % T[el][0])
thermo.write('# maxtemp:  %f\n' % T[el][-1])
thermo.write('# timestep: %f\n' % dt[units[el]])
thermo.write('# phmc:     %f\n' % p_hmc[el])
thermo.write('# lnvolmax: %f\n' % lnvol_max[el])
thermo.write('# ------------------------------------------------------------------------------------\n')
thermo.write('# | stp | temp | terr | pe | ke | virial | vol | acchmc | accvol | mdpehmv | mdpevol |\n')
thermo.write('# ------------------------------------------------------------------------------------\n')

# -------------------------------
# constant definitions
# -------------------------------

if units[el] == 'real':
    N_A = 6.0221409e23                               # avagadro number [num/mol]
    kB = 3.29983e-27                                 # boltzmann constant [kcal/K]
    R = kB*N_A                                       # gas constant [kcal/(mol K)]
    Et = R*T[el]                                     # thermal energy [kcal/mol]
    Pf = 1e-30*(1.01325e5*P[el])/(4.184e3*kB*T[el])  # metropolis prefactor [1/A^3]
if units[el] == 'metal':
    kB = 8.61733e-5                                  # boltzmann constant [eV/K]
    Et = natoms*kB*T[el]                             # thermal energy [eV]
    Pf = 1e-30*(1e5*P[el])/(1.60218e-19*kB*T[el])    # metropolis prefactor [1/A^3]
if units[el] == 'lj':
    # doesn't even stand a chance
    kB = 1.0
    Et = natoms*kB*T[el]
    Pf = P[el]/Et 

# -------------------
# npt-hmc monte carlo
# -------------------

# npt-hmc loop through temperatures
for i in xrange(len(T[el])):
    # initialize try and acceptation counters for npt-hmc
    nacchmc = 0
    ntryhmc = 0
    naccvol = 0
    ntryvol = 0
    # total potential energy change
    tdpehmc = 0
    tdpevol = 0
    # extract initial positions and velocities
    x = np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3))
    v = np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3))
    # extract initial energies
    pe = lmps.extract_compute('thermo_pe', None, 0)/Et[i]
    ke = lmps.extract_compute('thermo_ke', None, 0)/Et[i]
    virial = lmps.extract_compute('thermo_press', None, 0)
    temp = lmps.extract_compute('thermo_temp', None, 0)
    terr = np.abs((T[el][i]-temp)/T[el][i])
    # extract box dimensions
    boxmin = lmps.extract_global('boxlo', 1)
    boxmax = lmps.extract_global('boxhi', 1)
    box = boxmax-boxmin
    vol = np.power(box, 3)
    print('----------------------------------------------')
    print('NPT-HMC Run %d at Temperature %fK for %s' % (i, T[el][i], el))
    print('----------------------------------------------')
    print('| stp | temp | terr | pe | ke | virial | vol |')
    print('----------------------------------------------')
    print('%d %.4f %.4f %.4E %.4E %.4E %.4f' % (1, temp, terr, Et[i]*pe, Et[i]*ke, virial, vol))
    print('------------------------------------------------------------------------------------')
    print('| stp | temp | terr | pe | ke | virial | vol | acchmc | accvol | mdpehmv | mdpevol |')
    print('------------------------------------------------------------------------------------')
    # loop through hmc sweeps
    for j in xrange(n_swps):
        # hamiltonian monte carlo
        if np.random.rand() <= p_hmc[el]:
            # update hmc tries
            ntryhmc += 1
            # save current physical properties
            x = np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3))
            v = np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3))
            pe = lmps.extract_compute('thermo_pe', None, 0)/Et[i]
            ke = lmps.extract_compute('thermo_ke', None, 0)/Et[i]
            etot = pe+ke
            # set new atom velocities and run md
            lmps.command('velocity all create %f %d dist gaussian' % (T[el][i], np.random.randint(1, 2**16)))
            lmps.command('velocity all zero linear')
            lmps.command('velocity all zero angular')
            lmps.command('run %d' % n_stps)  # this part should be parallel
            # set new physical properties
            xnew = np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3))
            vnew = np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3))
            penew = lmps.extract_compute('thermo_pe', None, 0)/Et[i]
            kenew = lmps.extract_compute('thermo_ke', None, 0)/Et[i]
            etotnew = penew+kenew
            # calculate hamiltonian criterion
            dE = etotnew-etot
            if np.random.rand() <= np.min([1, np.exp(-dE)]):
                # update hamiltonian acceptations
                nacchmc += 1
                tdpehmc += penew-pe
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
        # npt monte carlo
        else:
            # update volume tries
            ntryvol += 1
            # save current physical properties
            x = np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3))
            pe = lmps.extract_compute('thermo_pe', None, 0)/Et[i]
            box = np.power(vol, 1.0/3.0)
            # save new physical properties
            lnvol = np.log(vol)+(np.random.rand()-0.5)*lnvol_max[el]
            volnew = np.exp(lnvol)
            boxnew = np.power(volnew, 1.0/3.0)
            scalef = boxnew/box
            xnew = boxnew/box*x
            # apply new physical properties
            lmps.command('change_box all x final 0.0 %.10f y final 0.0 %.10f z final 0.0 %.10f units box' % (3*(boxnew,)))
            lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
            lmps.command('run 0')
            penew = lmps.extract_compute('thermo_pe', None, 0)/Et[i]
            # calculate enthalpy criterion
            dH = (penew-pe)+Pf[i]*(volnew-vol)-(natoms+1.)*np.log(volnew/vol)
            if np.random.rand() <= np.min([1, np.exp(-dH)]):
                # update volume acceptations
                naccvol += 1
                tdpevol += penew-pe
                # save new physical properties
                pe = penew
                vol = volnew
                box = boxnew
                x = xnew
            else:
                # revert physical properties
                lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(x))
                lmps.command('change_box all x final 0.0 %.10f y final 0.0 %.10f z final 0.0 %.10f units box' % (3*(box,)))
                lmps.command('run 0')
        # data storage
        if (j+1) % mod == 0:
            # acceptance ratios
            acchmc = np.nan_to_num(np.float64(nacchmc)/ntryhmc)
            accvol = np.nan_to_num(np.float64(naccvol)/ntryvol)
            # mean mc arguments
            mdpehmc = np.nan_to_num(np.float64(tdpehmc)/nacchmc)
            mdpevol = np.nan_to_num(np.float64(tdpevol)/naccvol)
            # calculate physical properties
            virial = lmps.extract_compute('thermo_press', None, 0)
            temp = lmps.extract_compute('thermo_temp', None, 0)
            terr = np.abs((T[el][i]-temp)/T[el][i])
            # print thermal argument string
            therm_args = (j+1, temp, terr, Et[i]*pe, Et[i]*ke, virial, vol, acchmc, accvol, mdpehmc, mdpevol)
            print('%d %.4f %.4f %.4E %.4E %.4E %.4f %.4f %.4f %.4E %.4E' % therm_args)
            # write data to file
            thermo.write('%d %.4f %.4f %.4E %.4E %.4E %.4f %.4f %.4f %.4E %.4E\n' % therm_args)
            traj.write('%d %.4f\n' % (natoms, box))
            for k in xrange(natoms):
                traj.write('%.4f %.4f %.4f\n' % tuple(x[3*k:3*k+3]))
    # flush write buffer
    thermo.flush()
    traj.flush()
    
# close data storage files
thermo.close()
traj.close()