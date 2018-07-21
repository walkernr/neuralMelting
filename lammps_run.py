# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:38:03 2018

@author: Nicholas
"""

from __future__ import division, print_function
import sys, subprocess

# verbosity
if '--verbose' in sys.argv:
    verbose = True
else:
    verbose = False
# serial mode
if '--serial' in sys.argv:
    parallel = False
else:
    parallel = True
# multithreading
if '--threading' in sys.argv:
    processes = False
else:
    processes = True
# turn off simulation
if '--nosim' in sys.argv:
    sim = False
else:
    sim = True
# turn of post-processing
if '--nopost' in sys.argv:
    post = False
else:
    post = True

# parallel arguments
nworker = 16
nthread = 1
# simulation name
name = 'test'
# element and system size
el = 'LJ'
sz = 4
# pressures
npress = 8
lpress = 1
hpress = 8
# temperatures
ntemp = 96
ltemp = 0.25
htemp = 2.5
# monte carlo parameters
cutoff = 1024
nsmpl = 1024
mod = 128
ppos = 0.015625
pvol = 0.25
nstps = 8
# command line argument list
cmd_args = ['--nworker', str(nworker),
            '--nthread', str(nthread),
            '--name', name,
            '--element', el,
            '--size', str(sz),
            '--npress', str(npress),
            '--rpress', str(lpress), str(hpress),
            '--ntemp', str(ntemp),
            '--rtemp', str(ltemp), str(htemp),
            '--cutoff', str(cutoff),
            '--nsmpl', str(nsmpl),
            '--mod', str(mod),
            '--ppos', str(ppos),
            '--pvol', str(pvol),
            '--nstps', str(nstps)]
# additional arguments
if verbose:
    cmd_args = cmd_args+['--verbose']
if not parallel:
    cmd_args = cmd_args+['--serial']
if not processes:
    cmd_args = cmd_args+['--threading']
# run simulation
if sim:
    subprocess.call(['python', 'lammps_remcmc.py']+cmd_args)
# run post-processing
if post:
    for i in xrange(npress):
        subprocess.call(['python', 'lammps_parse.py']+cmd_args+['--pressindex', str(i)])
    for i in xrange(npress):
        subprocess.call(['python', 'lammps_rdf.py']+cmd_args+['--pressindex', str(i)])