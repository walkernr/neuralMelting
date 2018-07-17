# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:38:03 2018

@author: Nicholas
"""

from __future__ import division, print_function
import subprocess

verbose = False

nworkers = 16
nthreads = 1

name = 'remcmc01'
el = 'LJ'
sz = 4

npress = 8
lpress = 1
hpress = 8

ntemp = 48
ltemp = 0.25
htemp = 2.5

cutoff = 1024
nsmpl = 1024
mod = 128
ppos = 0.015625
pvol = 0.25
nstps = 16

cmd_flags = ['--nworkers', str(nworkers),
             '--nthreads', str(nthreads),
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
if verbose:
    cmd_args = cmd_args+['--verbose']

subprocess.call(['python', 'lammps_remcmc.py']+cmd_flags)
for i in xrange(npress):
    subprocess.call(['python', 'lammps_parse.py']+cmd_flags+['--pressindex', str(i)])
for i in xrange(npress):
    subprocess.call(['python', 'lammps_rdf.py']+cmd_flags+['--pressindex', str(i)])