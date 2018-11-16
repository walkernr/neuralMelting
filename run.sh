mkdir ./output/remcmc_3
cd ./output/remcmc_3

python ../../scripts/lammps_remcmc.py -v -p -c -nw 16 -nt 1 -mt fork -n remcmc_init_3 -ss 3 -sc 1024
rm -r ./dask-worker-space
python ../../scripts/lammps_remcmc.py -v -p -c -nw 16 -nt 1 -mt fork -rn remcmc_init_3 -n remcmc_run_3 -ss 3
rm -r ./dask-worker-space
for i in {0..7}
do
  python ../../scripts/lammps_parse.py -v -n remcmc_run_3 -i $i
  python ../../scripts/lammps_rdf.py -v -p -c -nw 16 -nt 1 -n remcmc_run_3 -i $i
  rm -r dask-worker-space
  python ../../scripts/lammps_neural.py -v -p -pt -n remcmc_run_3 -i $i -ep 2
  python ../../scripts/lammps_cluster.py -v -pt -n remcmc_run_3 -i $i
done
python ../../scripts/lammps_post.py -v -n remcmc_run_3

