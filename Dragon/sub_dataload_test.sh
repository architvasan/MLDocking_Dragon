#!/bin/bash -l
#PBS -N ml_docking_dragon
#PBS -l walltime=3:00:00
#PBS -l select=32
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -A hpe_dragon_collab
#PBS -q prod
#PBS -V

cd $PBS_O_WORKDIR
RUN_SCRIPT="./run_driver_loadonly.sh"
NUM_FILES="8192 32768 131072 500354"
for n in $NUM_FILES; do
    mkdir $n"_32chunks"
    cp $RUN_SCRIPT $n"_32chunks"
    cd $n"_32chunks"
    timeout 10m $RUN_SCRIPT $n 32 >>test.out 2>&1
    cd ..
    echo ""
    sleep 60
    
    mkdir $n"_8chunks"
    cp $RUN_SCRIPT $n"_8chunks"
    cd $n"_8chunks"
    timeout 10m $RUN_SCRIPT $n 8 >>test.out 2>&1
    cd ..
    echo ""
    sleep 60

    mkdir $n"_1chunks"
    cp $RUN_SCRIPT $n"_1chunks"
    cd $n"_1chunks"
    timeout 10m $RUN_SCRIPT $n 1 >>test.out 2>&1
    cd ..
    echo ""
    sleep 60
done




