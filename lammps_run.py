# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:38:03 2018

@author: Nicholas
"""

from __future__ import division, print_function
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
parser.add_argument('-p', '--parallel', help='parallel run', action='store_true')
parser.add_argument('-d', '--distributed', help='distributed run', action='store_true')
parser.add_argument('-s', '--simulation', help='simulation run', action='store_true')
parser.add_argument('-r', '--rdf', help='radial distribution run', action='store_true')

args = parser.parse_args()

VERBOSE = args.verbose
PARALLEL = args.parallel
DISTRIBUTED = args.distributed
SIM = args.simulation
RDF = args.rdf

SIM_ARGS = []
PRS_ARGS = []
RDF_ARGS = []

if VERBOSE:
    if SIM:
        SIM_ARGS.append('-v')
    if RDF:
        PRS_ARGS.append('-v')
        RDF_ARGS.append('-v')
if PARALLEL:
    if SIM:
        SIM_ARGS.append('-p')
        if DISTRIBUTED:
            SIM_ARGS.append('-d')
    if RDF:
        RDF_ARGS.append('-p')
        if DISTRIBUTED:
            RDF_ARGS.append('-d')

QUEUE = 'lasigma'
ALLOC = 'hpc_lasigma01'
NODES = 1
PPN = 16
WALLTIME = 72
MEM = NODES*32
if DISTRIBUTED:
    NTHREAD = 1
    NWORKER = int(NODES*PPN/NTHREAD)
else:
    NTHREAD = 1
    NWORKER = int(PPN/NTHREAD)
NAME = 'test'
EL = 'LJ'
SZ = 4
NPRESS = 4
LPRESS, HPRESS = (2.0, 8.0)
NTEMP = 48
LTEMP, HTEMP = (0.25, 2.5)
CUTOFF = 1024
NSMPL = 1024
MOD = 128
PPOS = 0.015625
PVOL = 0.25
NSTPS = 8

if PARALLEL:
    par_args = ['-nw', str(NWORKER),
                '-nt', str(NTHREAD)]
    if DISTRIBUTED:
        par_args = par_args+['-q', QUEUE,
                             '-a', ALLOC,
                             '-nn', str(NODES),
                             '-np', str(PPN),
                             '-w', str(WALLTIME),
                             '-m', str(MEM)]
    if SIM:
        SIM_ARGS = SIM_ARGS+par_args
    if RDF:
        RDF_ARGS = RDF_ARGS+par_args

id_args = ['-n', NAME,
           '-e', EL]
if SIM:
    SIM_ARGS = SIM_ARGS+id_args+['-ss', str(SZ),
                                 '-pn', str(NPRESS),
                                 '-pr', str(LPRESS), str(HPRESS),
                                 '-tn', str(NTEMP),
                                 '-tr', str(LTEMP), str(HTEMP),
                                 '-sc', str(CUTOFF),
                                 '-sn', str(NSMPL),
                                 '-sm', str(MOD),
                                 '-pm', str(PPOS),
                                 '-vm', str(PVOL),
                                 '-t', str(NSTPS)]
if RDF:
    RDF_ARGS = RDF_ARGS+id_args
    PRS_ARGS = PRS_ARGS+id_args

if SIM:
    subprocess.call(['python', 'lammps_remcmc.py']+SIM_ARGS)
if RDF:
    for i in xrange(NPRESS):
        subprocess.call(['python', 'lammps_parse.py']+PRS_ARGS+['-i', str(i)])
    for i in xrange(NPRESS):
        subprocess.call(['python', 'lammps_rdf.py']+RDF_ARGS+['-i', str(i)])
