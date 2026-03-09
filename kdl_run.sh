#!/bin/bash -l

# Setup
#cd $PBS_O_WORKDIR
#NODES=$(cat $PBS_NODEFILE | wc -l)

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
	source /flare/datascience/csimpson/hpe_dragon_collab/env.sh
	export RECEPTOR_FILE=/flare/datascience/dragon/receptor_files/3clpro_7bqy.oedu
	DATA_PATH=/flare/datascience/dragon/tiny
	export DRIVER_PATH=/flare/datascience/csimpson/hpe_dragon_collab/MLDocking_Dragon/Dragon/
	;;
    *"polaris"* )
	POLARIS=1
	echo "Setting up for Polaris run"
	source /eagle/hpe_dragon_collab/csimpson/env.sh
	DATA_PATH=/eagle/hpe_dragon_collab/csimpson/ZINC-22-presorted/tiny
	export DRIVER_PATH=/eagle/hpe_dragon_collab/csimpson/MLDocking_Dragon/Dragon/
	export RECEPTOR_FILE=/eagle/hpe_dragon_collab/avasan/3clpro_7bqy.oedu
	;;
	*"pinoak"* )
	PINOAK=1
	echo "Setting up for Pinoak run"
	#source /eagle/hpe_dragon_collab/csimpson/env.sh
	DATA_PATH=/lus/scratch/mendygra/alcf/MLDocking_Dragon/tiny
	export DRIVER_PATH=/home/users/klee/home/Repos/fix/MLDocking_Dragon/Dragon/
	export RECEPTOR_FILE=/eagle/hpe_dragon_collab/avasan/3clpro_7bqy.oedu
	;;
	*"hotlum"* )
	HOTLUM=1
	echo "Setting up for Hotlum run"
	#source /eagle/hpe_dragon_collab/csimpson/env.sh
	DATA_PATH=/lus/scratch/wahlc/dragon/collabs/anl/data/med
	export DRIVER_PATH=/home/users/klee/home/Repos/MLDocking/MLDocking_Dragon/Dragon/
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
    export CPU_AFFINITY="list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:52-59,156-163:60-67,164-171:68-75,172-179:76-83,180-187:84-91,188-195:92-99,196-203"
    export ZEX_NUMBER_OF_CCS=0:1,1:1,2:1,3:1,4:1,5:1
    # Other machine specific env vars
    # Some env vars from Archit's script
    export PLATFORM_NUM_GPU_TILES=2
    export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
    export ITEX_LIMIT_MEMORY_SIZE_IN_MB=8192
    export ITEX_ENABLE_NEXTPLUGGABLE_DEVICE=0
    PROCS_PER_NODE=104
    MEM_PER_NODE=256
fi
if [[ -n $POLARIS || -n $SIRIUS || -n $PINOAK || -n $HOTLUM ]]; then
    echo "Setting up for Nvidia GPUS"
    export GPU_DEVICES="3,2,1,0"
    export CPU_AFFINITY="list:0-7,32-39:8-15,40-47:16-23,48-55:24-31,56-63"
    export USE_MPI_SORT=1
    MAX_PROCS=256
    TOTAL_DDICT_SIZE=2048 # GB
fi

echo "Location of dragon:"
which dragon

# Dragon env vars
#export FI_MR_CACHE_MAX_COUNT=0
#export FI_CXI_ODP=1
#export DRAGON_HSTA_NO_NET_CONFIG=1
#export DRAGON_DEFAULT_SEG_SZ=34359738368

export PYTHONPATH=$DRIVER_PATH:$PYTHONPATH

MANAGERS_PER_NODE=1
NUM_NODES=32
echo Running on $NODES nodes
echo Reading files from $DATA_PATH
echo Running with $MAX_PROCS max. processes across all nodes
echo Total DDict Size $TOTAL_DDICT_SIZE
echo Number of DDict Managers $MANAGERS

# Set debug flag
#DEBUG_STR="-l dragon_file=DEBUG -l actor_file=DEBUG"
DEBUG_STR=
#DEBUG_STR="-l DEBUG"

# Run
module list
echo $LD_LIBRARY_PATH
dragon-cleanup
EXE="dragon $DEBUG_STR ${DRIVER_PATH}/data_loader/data_loader_presorted.py
--managers_per_node=$MANAGERS_PER_NODE \
--num_nodes=$NUM_NODES
--data_path=${DATA_PATH} \
--max_procs=$MAX_PROCS \
--total_mem_size=$TOTAL_DDICT_SIZE"

echo $EXE
${EXE}
