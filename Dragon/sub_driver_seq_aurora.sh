#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N ml_docking_dragon
#PBS -l walltime=01:00:00
#PBS -l select=2
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -A hpe_dragon_collab
#PBS -q debug-scaling
#PBS -V


cd $PBS_O_WORKDIR
rm -r *log*
rm ddict*
./run_driver_seq.sh 256 # Test dataset run
# ./run_driver_seq.sh 500354 # Full dataset run
