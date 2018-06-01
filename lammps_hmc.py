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
n_swps = 1024*mod     # total hmc sweeps
n_stps = 16           # md steps during hmc
seed = 256            # random seed
phmc = 0.75           # probability of hmc move
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
dbox = {'Ti': 0.0009765625*lat['Ti'][1],
        'Al': 0.0009765625*lat['Al'][1],
        'Ni': 0.0009765625*lat['Ni'][1],
        'Cu': 0.0009765625*lat['Cu'][1],
        'LJ': 0.0009765625*lat['LJ'][1]}
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
    lmpsfile.write('atom_modify map yes\n\n')
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
        lmpsfile.write('pair_coeff 1 1 %f %f 2.5\n\n' % lj_param)
    # minimize lattice structure
    lmpsfile.write('fix 1 all box/relax iso %f vmax %f\n' % (P[el], 0.0009765625))
    lmpsfile.write('minimize 0.0 %f %d %d\n' % (1.49011612e-8, 1024, 8192))
    lmpsfile.write('unfix 1\n\n')
    # compute kinetic energy
    lmpsfile.write('compute thermo_ke all ke\n\n')
    # dynamics initialization
    lmpsfile.write('timestep %f\n' % dt[units[el]])
    lmpsfile.write('fix 1 all nve\n')
    lmpsfile.write('run 0')
    # close file and return name
    lmpsfile.close()
    return lmpsfilein

# ---------------------
# initialize lammps run
# ---------------------

# generate input file
lmpsfilein = lammps_input()
# initialize lammps
lmps = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
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
thermo.write('# phmc:     %f\n' % phmc)
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
thermo.write('# dboxmax:  %f\n' % dbox[el])
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
    Et = kB*T[el]                                    # thermal energy [eV]
    Pf = 1e-30*(1e5*P[el])/(1.60218e-19*kB*T[el])    # metropolis prefactor [1/A^3]
if units[el] == 'lj':
    kB = 1.0                                         # boltzmann constant (normalized and unitless)
    Et = kB*T[el]                                    # thermal energy [T*]
    Pf = P[el]/(kB*T[el])                            # metropolis prefactor [1/r*^3]

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
    x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
    v = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3)))
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
        if np.random.rand() <= phmc:
            # update hmc tries
            ntryhmc += 1
            # save current physical properties
            x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
            v = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3)))
            pe = lmps.extract_compute('thermo_pe', None, 0)/Et[i]
            ke = lmps.extract_compute('thermo_ke', None, 0)/Et[i]
            etot = pe+ke
            # set new atom velocities and run md
            lmps.command('velocity all create %f %d dist gaussian' % (T[el][i], np.random.randint(1, 2**16)))
            lmps.command('velocity all zero linear')
            lmps.command('velocity all zero angular')
            # lmps.command('timestep %f' % dt[units[el]])
            lmps.command('run %d' % n_stps)  # this part should be parallel
            # set new physical properties
            xnew = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
            vnew = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('v', 1, 3)))
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
        # npt monte carlo
        else:
            # update volume tries
            ntryvol += 1
            # save current physical properties
            boxmin = lmps.extract_global('boxlo', 1)
            boxmax = lmps.extract_global('boxhi', 1)
            box = boxmax-boxmin
            vol = np.power(box, 3)
            x = np.copy(np.ctypeslib.as_array(lmps.gather_atoms('x', 1, 3)))
            pe = lmps.extract_compute('thermo_pe', None, 0)/Et[i]
            # save new physical properties
            boxnew = box+(np.random.rand()-0.5)*dbox[el]
            volnew = np.power(boxnew, 3)
            scalef = boxnew/box
            xnew = scalef*x
            # apply new physical properties
            lmps.command('change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f units box' % (3*(boxnew,)))
            lmps.scatter_atoms('x', 1, 3, np.ctypeslib.as_ctypes(xnew))
            lmps.command('run 0')
            penew = lmps.extract_compute('thermo_pe', None, 0)/Et[i]
            # calculate enthalpy criterion
            dH = (penew-pe)+Pf[i]*(volnew-vol)-natoms*np.log(volnew/vol)
            if np.random.rand() <= np.min([1, np.exp(-dH)]):
                # update volume acceptations
                naccvol += 1
                tdpevol += penew-pe
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
        # timestep and max box adjustment update; data storage
        if (j+1) % mod == 0:
            # acceptance ratios
            acchmc = np.nan_to_num(np.float64(nacchmc)/np.float64(ntryhmc))
            accvol = np.nan_to_num(np.float64(naccvol)/np.float64(ntryvol))
            # update timestep and max box adjustment
            # if acchmc < 0.5:
                # dt[units[el]] *= 0.9375
            # else:
                # dt[units[el]] *= 1.0625
            if accvol < 0.5:
                dbox[el] *= 0.9375
            else:
                dbox[el] *= 1.0625
            # mean potential energy changes
            mdpehmc = np.nan_to_num(np.float64(tdpehmc)/np.float64(nacchmc))
            mdpevol = np.nan_to_num(np.float64(tdpevol)/np.float64(naccvol))
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
#close lammps
lmps.close()