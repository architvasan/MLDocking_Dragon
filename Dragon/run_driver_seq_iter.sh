#!/bin/bash -l

# Setup
#cd $PBS_O_WORKDIR
NODES=$(cat $PBS_NODEFILE | wc -l)

# Determine which machine you are on
FULL_HOSTNAME=`hostname -f`
echo "FULL_HOSTNAME="$FULL_HOSTNAME

case "$FULL_HOSTNAME" in
    *"americas"* )
	SUNSPOT=1
	echo "Setting up for Sunspot run"
	source /gila/Aurora_deployment/csimpson/hpe_dragon_collab/env.sh
	export RECEPTOR_FILE=/gila/Aurora_deployment/dragon/receptor_files/3clpro_7bqy.oedu
	DATA_PATH=/gila/Aurora_deployment/dragon/data/tiny
	export DRIVER_PATH=/gila/Aurora_deployment/csimpson/hpe_dragon_collab/MLDocking_Dragon/Dragon/
	;;
    *"aurora"* )
	AURORA=1
	echo "Setting up for Aurora run"
	source /flare/hpe_dragon_collab/balin/PASC25/env.sh
	export RECEPTOR_FILE=/flare/datascience/dragon/receptor_files/3clpro_7bqy.oedu
	DATA_PATH=/flare/datascience/dragon/tiny
	export DRIVER_PATH=/flare/hpe_dragon_collab/balin/PASC25/MLDocking_Dragon/Dragon/
	export OE_LICENSE=/flare/hpe_dragon_collab/balin/oe_license.txt
    ;;
    *"polaris"* )
	POLARIS=1
	echo "Setting up for Polaris run"
	source /eagle/hpe_dragon_collab/csimpson/env.sh
	DATA_PATH=/eagle/hpe_dragon_collab/csimpson/ZINC-22-presorted/tiny
	export DRIVER_PATH=/eagle/hpe_dragon_collab/csimpson/MLDocking_Dragon/Dragon/
	export RECEPTOR_FILE=/eagle/hpe_dragon_collab/avasan/3clpro_7bqy.oedu
	;;
    *"sirius"* )
	SIRIUS=1
	echo "Setting up for Sirius run"
	# TODO: add other paths and environment
	export RECEPTOR_FILE=/home/csimpson/openeye/3clpro_7bqy.oedu
	;;
esac

echo "Setting hardware affinities"
# Set gpu and cpu affinities and other machine specific env vars
if [[ -n $SUNSPOT || -n $AURORA ]]; then
    echo "Setting up for Intel GPUS"
    export GPU_DEVICES="0.0,0.1,1.0,1.1,2.0,2.1,3.0,3.1,4.0,4.1,5.0,5.1"
    export USE_CCS=0 #1 for on
    export ZEX_NUMBER_OF_CCS=0:1,1:1,2:1,3:1,4:1,5:1
    export DATA_DD_CPU_AFFINITY="44,45,46,47,96,97,98,99"
    export SIM_DD_CPU_AFFINITY="44,45,46,47,96,97,98,99"
    export MODEL_DD_CPU_AFFINITY="48,49,50,51,100,101,102,103"
    export INF_CPU_AFFINITY="1-4:5-8:9-12:13-16:17-20:21-24:53-56:57-60:61-64:65-68:69-72:73-76"
    #export INF_CPU_AFFINITY="1,2:3,4:5,6:7,8:9,10:11,12:13,14:15,16:17,18:19,20:21,22:23,24:53,54:55,56:57,58:59,60:61,62:63,64:65,66:67,68:69,70:71,72:73,74:75,76"
    export SORT_CPU_AFFINITY="" # only needed for MPI sort
    export SIM_CPU_AFFINITY="5-8:9-12:13-16:17-20:21-24:53-56:57-60:61-64:65-68:69-72:73-76"
    export TRAIN_CPU_AFFINITY="1,2,3,4"
    export SKIP_THREADS="0,52,104,156"
    # Other machine specific env vars
    # Some env vars from Archit's script
    export PLATFORM_NUM_GPU_TILES=2
    export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
    export PROCS_PER_NODE=104
    export MEM_PER_NODE=256
    #export TF_CPP_MAX_VLOG_LEVEL=3
    export ITEX_VERBOSE=0
    export TF_CPP_MIN_LOG_LEVEL=3
    export ITEX_CPP_MIN_LOG_LEVEL=2
    export PYTHONWARNINGS="ignore::FutureWarning"
    export ITEX_LIMIT_MEMORY_SIZE_IN_MB=8192
    export ITEX_ENABLE_NEXTPLUGGABLE_DEVICE=0
fi
if [[ -n $POLARIS || -n $SIRIUS ]]; then
    echo "Setting up for Nvidia GPUS"
    export GPU_DEVICES="3,2,1,0"
    export CPU_AFFINITY="list:0-7,32-39:8-15,40-47:16-23,48-55:24-31,56-63"
    export USE_MPI_SORT=1
    PROCS_PER_NODE=32
    MEM_PER_NODE=128
fi

echo "Location of dragon:"
which dragon

# Dragon env vars
# None

# Other env vars
# for tensorflow reporting
export OMP_NUM_THREADS=1
export PYTHONPATH=$DRIVER_PATH:$PYTHONPATH

MANAGERS=1
echo Running on $NODES nodes
echo Reading files from $DATA_PATH
echo Running with $PROCS_PER_NODE max. processes in Pool per Node
echo Mem per node $MEM_PER_NODE
echo Managers $MANAGERS

# Set debug flag
#DEBUG_STR="-l dragon_file=DEBUG -l actor_file=DEBUG"
DEBUG_STR=
#DEBUG_STR="-l DEBUG"

# Run
module list
echo $LD_LIBRARY_PATH
#dragon-cleanup
EXE="dragon $DEBUG_STR ${DRIVER_PATH}/dragon_driver_sequential.py \
--max_iter=2 --candidate_fraction=0.1 \
--logging=debug \
--managers_per_node=$MANAGERS \
--data_path=${DATA_PATH} \
--max_procs_per_node=$PROCS_PER_NODE \
--mem_per_node=$MEM_PER_NODE \
--inference_node_num=2 --sorting_node_num=2 --simulation_node_num=1"

echo $EXE
echo
${EXE}
