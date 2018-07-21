# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:38:03 2018

@author: Nicholas
"""

from __future__ import division, print_function
import argparse, subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
parser.add_argument('-p', '--parallel', help='parallel run', action='store_true')
parser.add_argument('-d', '--distributed', help='distributed run', action='store_true')
parser.add_argument('-s', '--simulation', help='simulation run', action='store_true')
parser.add_argument('-r', '--rdf', help='radial distribution run', action='store_true')

args = parser.parse_args()

verbose = args.verbose
parallel = args.parallel
distributed = args.distributed
sim = args.simulation
rdf = args.rdf

sim_args = []
prs_args = []
rdf_args = []
if verbose:
    if sim:
        sim_args.append('-v')
    if rdf:
        prs_args.append('-v')
        rdf_args.append('-v')
if parallel:
    if sim:
        sim_args.append('-p')
        if distributed:
            sim_args.append('-d')
    if rdf:
        rdf_args.append('-p')
        if distributed:
            rdf_args.append('-d')
            
queue = 'lasigma'
alloc = 'hpc_lasigma01'
nodes = 1
ppn = 16
walltime = 72
mem = 32
nworker = 16
nthread = 1
name = 'test'
el = 'LJ'
sz = 4
npress = 4
lpress, hpress = (2.0, 8.0)
ntemp = 16
ltemp, htemp = (0.25, 2.5)
cutoff = 16
nsmpl = 16
mod = 16
ppos = 0.015625
pvol = 0.25
nstps = 8

if parallel:
    par_args = ['-nw', str(nworker),
                '-nt', str(nthread)]
    if distributed:
        par_args = par_args+['-q', queue,
                             '-a', alloc,
                             '-nn', str(nodes),
                             '-np', str(ppn),
                             '-w', str(walltime),
                             '-m', str(mem)]
    if sim:
        sim_args = sim_args+par_args
    if rdf:
        rdf_args = rdf_args+par_args

id_args = ['-n', name,
           '-e', el,
           '-pn', str(npress),
           '-pr', str(lpress), str(hpress)]
if sim:
    sim_args = sim_args+id_args+['-ss', str(sz),
                                 '-tn', str(ntemp),
                                 '-tr', str(ltemp), str(htemp),
                                 '-sc', str(cutoff),
                                 '-sn', str(nsmpl),
                                 '-sm', str(mod),
                                 '-pm', str(ppos),
                                 '-vm', str(pvol),
                                 '-t', str(nstps)]
if rdf:
    rdf_args = rdf_args+id_args
    prs_args = prs_args+id_args
                         
if sim:
    subprocess.call(['python', 'lammps_remcmc.py']+sim_args)
if rdf:
    for i in xrange(npress):
        subprocess.call(['python', 'lammps_parse.py']+prs_args+['-i', str(i)])
    for i in xrange(npress):
        subprocess.call(['python', 'lammps_rdf.py']+rdf_args+['-i', str(i)])