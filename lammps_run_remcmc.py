# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:38:03 2018

@author: Nicholas
"""

from __future__ import division, print_function
import subprocess

nworkers = 16
nthreads = 1

name = 'test'
el = 'LJ'
sz = 4

npress = 4
lpress = 1
hpress = 8

ntemp = 4
ltemp = 0.25
htemp = 2.5

cutoff = 4
nsmpl = 4
mod = 4
ppos = 0.015625
pvol = 0.25
nstps = 4

cmd_flags = ['--verbose',
             '--nworkers', str(nworkers),
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

subprocess.call(['python', 'lammps_remcmc_distributed.py']+cmd_flags)
for i in xrange(npress):
    subprocess.call(['python', 'lammps_parse.py']+cmd_flags+['--pressindex', str(i)])
for i in xrange(npress):
    subprocess.call(['python', 'lammps_rdf_distributed.py']+cmd_flags+['--pressindex', str(i)])