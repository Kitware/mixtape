#!/bin/bash

#PBS -A ARLAP5015C500
#PBS -l select=2:ncpus=128:mpiprocs=128
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -j oe
#PBS -N out
#PBS -S /bin/bash

# =============================================================================
# Set up Conda environment
# =============================================================================

# Get the base directory of the Conda installation.
CONDA_BASE=$(conda info --base)
# Enable `conda` commands.
source $CONDA_BASE/etc/profile.d/conda.sh
# Activate the "ray" Conda environment, which contains the required dependencies.
conda activate ray


# =============================================================================
# Prepare host files
# =============================================================================

# Reads the list of allocated nodes (`$PBS_NODEFILE`), sorts them alphabetically,
# and saves the result to `local_hostfile`. This ensures node order consistency.
cat $PBS_NODEFILE | sort > local_hostfile
# Count the total number of nodes in the `local_hostfile` and stores the result
# in `num_nodes`.
num_nodes=$(wc -l local_hostfile | awk '{print $1}')
# Calculate the number of worker nodes by subtracting 1 from `num_nodes`.
# The first node will act as the master/head node.
workers=$(python -c "print(int(int(${num_nodes})-1))")
# Extract the last `$workers` entries from `local_hostfile` and save them
# to `worker_hosts`, which contains the worker node hostnames.
tail -n $workers local_hostfile > worker_hosts
# Extract the first entry from `local_hostfile` and save it to `master_hosts`,
# which contains the hostname of the master/head node.
head -n 1 local_hostfile > master_hosts


# =============================================================================
# Start the Ray head node
# =============================================================================

# Start the Ray head node on the master node. 
#   node-ip-address: Specifies the IP of the head node (from `master_hosts`).
#   port: Defines the port for communication.
#   num-cpus: Allocates all 128 CPUs on the head node to Ray.
#   num-gpus: Specifies that no GPUs are used.
#   temp-dir: Sets the temporary directory.
#   block: Ensures the Ray process remains running in the foreground.

ray start --head \
    --node-ip-address=$(head -n 1 master_hosts) \
    --port=29500 \
    --num-cpus 128 \
    --num-gpus 0 \
    --temp-dir=/tmp \
    --block &

# Wait 15 seconds to allow the head node to fully initialize.
sleep 15


# =============================================================================
# Start the Ray worker nodes
# =============================================================================

# Start Ray worker nodes using `mpirun`, with one process per worker node.
#   hostfile worker_hosts: Specifies the worker node hostnames.
#   address: Connects the worker nodes to the head node at the specified address.
#   num-cpus: Allocates all 128 CPUs on each worker node.
#   num-gpus: Specifies that no GPUs are used.
#   block: Ensures each worker process remains running in the foreground.

mpirun -np 1 --hostfile worker_hosts \
    ray start --address=$(head -n 1 master_hosts):29500 \
    --num-cpus 128 \
    --num-gpus 0 \
    --block &

# Wait 15 seconds to allow the worker nodes to connect to the head node.
sleep 15


# =============================================================================
# Run the user script
# =============================================================================

# Execute the `test.py` Python script
python -u test.py


# =============================================================================
# Cleanup
# =============================================================================

# Stops all Ray processes on the current node.
ray stop
