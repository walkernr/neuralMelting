s=2
# make and move to directory for output data
mkdir ./output/remcmc_$s
cd ./output/remcmc_$s

# remcmc equilibration run
python ../../scripts/lammps_remcmc.py -v -c -nw 16 -nt 1 -mt fork -n remcmc_init_$s -ss $s -pn 32 -tn 32 -sc 1024
rm -r ./dask-worker-space
# remcmc data collection run
python ../../scripts/lammps_remcmc.py -v -c -nw 16 -nt 1 -mt fork -rn remcmc_init_$s -n remcmc_run_$s -ss $s -pn 32 -tn 32
rm -r ./dask-worker-space
# parse output
python ../../scripts/lammps_parse.py -v -n remcmc_run_$s
# calculate structural data
python ../../scripts/lammps_distr.py -v -c -nw 16 -nt 1 -n remcmc_run_$s
rm -r dask-worker-space