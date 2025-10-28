#!/bin/bash -l

# Maximum number of files: 500354

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
	#module load copper
	#launch_copper.sh
	source /flare/hpe_dragon_collab/csimpson/env.sh
	export RECEPTOR_FILE=/flare/datascience/dragon/receptor_files/3clpro_7bqy.oedu
	DATA_PATH=/flare/datascience/dragon/ZINC-22-2D-smaller_files
	export DRIVER_PATH=/flare/hpe_dragon_collab/csimpson/MLDocking_Dragon/Dragon/
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
    export CPU_AFFINITY="list:1-7,105-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:53-59,157-163:60-67,164-171:68-75,172-179:76-83,180-187:84-91,188-195:92-99,196-203"
    export SKIP_THREADS="0,52,104,156"
    export ZEX_NUMBER_OF_CCS=0:1,1:1,2:1,3:1,4:1,5:1
    # Other machine specific env vars
    # Some env vars from Archit's script
    export PLATFORM_NUM_GPU_TILES=2
    export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
    export ITEX_LIMIT_MEMORY_SIZE_IN_MB=8192
    export ITEX_ENABLE_NEXTPLUGGABLE_DEVICE=0
    PROCS_PER_NODE=104
    MEM_PER_NODE=512
fi
if [[ -n $POLARIS || -n $SIRIUS ]]; then
    echo "Setting up for Nvidia GPUS"
    export GPU_DEVICES="3,2,1,0"
    export CPU_AFFINITY="list:0-7,32-39:8-15,40-47:16-23,48-55:24-31,56-63"
    export USE_MPI_SORT=1
    PROCS_PER_NODE=32
    MEM_PER_NODE=256
fi

echo "Location of dragon:"
which dragon

# Clean up dragon processes
dragon-cleanup
sleep 10

# Dragon env vars
# None

# Other env vars
# for tensorflow reporting
export OMP_NUM_THREADS=1
export ITEX_VERBOSE=0
export TF_CPP_MIN_LOG_LEVEL=3
export ITEX_CPP_MIN_LOG_LEVEL=2
export PYTHONWARNINGS="ignore::FutureWarning"
export PYTHONPATH=$DRIVER_PATH:$PYTHONPATH
#export USE_MPI_SORT=1

export DRAGON_DEFAULT_SEG_SIZE=17179869184

MANAGERS=1
#FILE_CHUNK_NUM=4
echo Running on $NODES nodes
echo Reading files from $DATA_PATH
echo Running with $PROCS_PER_NODE max. processes in Pool per Node
echo Mem per node $MEM_PER_NODE
echo Managers $MANAGERS
echo Number of files: $1

# Set debug flag
#DEBUG_STR="-l dragon_file=DEBUG -l actor_file=DEBUG"
DEBUG_STR=
#DEBUG_STR="-l DEBUG"

# Run
module list
echo $LD_LIBRARY_PATH
#dragon-cleanup
EXE="dragon $DEBUG_STR ${DRIVER_PATH}/dragon_driver_loadonly.py \
--managers_per_node=$MANAGERS \
--data_path=${DATA_PATH} \
--max_procs_per_node=$PROCS_PER_NODE \
--mem_per_node=$MEM_PER_NODE \
--file_chunk_num=$2 \
--num_files $1"

echo $EXE
${EXE}
