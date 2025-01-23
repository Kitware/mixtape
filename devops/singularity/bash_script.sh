#PBS -S /bin/bash
#PBS -l walltime=00:10:00
#PBS -A {{ PROJECT ID }}
#PBS -j oe
#PBS -l select=1:ncpus=128:mpiprocs=2:ngpus=1
#PBS -q debug
#PBS -N {{ JOB NAME }}
#PBS -m e
#PBS -M {{ EMAIL ADDRESS }}

# =============================================================================
# Load environment modules
# =============================================================================

# Loads the loop module, required for mounting loop devices (like Singularity images)
modprobe loop
# Initialize environment for modules
source $MODULESHOME/init/bash
# Add the Singularity module to the environment
module add singularity/3.8.4
# Add the NVIDIA module for GPU support
module add nvidia/22.3

# =============================================================================
# Setup
# =============================================================================

# Build the API URL to communicate with the server
PORT="{{ 8000 }}"
API_URL="http://localhost:$PORT"
echo "API URL: $API_URL"

# Change the directory to where the job was submitted from
cd ${PBS_O_WORKDIR}
# Set Singularity's temp directory to the job's TMPDIR
export SINGULARITY_TMPDIR=$TMPDIR

# =============================================================================
# Start the server
# =============================================================================

# Execute the Singularity container
# Bind the `logs` directory
# Run FastAPI on the specified port using uvicorn
singularity exec \
--bind {{  ${PBS_O_WORKDIR}/logs:/app/logs  }} \
${PBS_O_WORKDIR}/mixtape.sif \
bash -c "PYTHONPATH=/app uvicorn src.main:app --host 0.0.0.0 --port $PORT" &

# Wait for the FastAPI service to be ready
WAIT_TIME={{ 5 }}
MAX_RETRIES={{ 24 }}
echo "Waiting for the FastAPI endpoint to become available..."
RETRIES=0
while ! curl -s "$API_URL" > /dev/null; do
    sleep "$WAIT_TIME"
    # Wait for the server to respond, retry if it fails
    ((RETRIES++))
    if [ "$RETRIES" -ge "$MAX_RETRIES" ]; then
        echo "FastAPI endpoint not available after $((WAIT_TIME * MAX_RETRIES)) seconds."
        singularity instance stop
        exit 1
    fi
    echo "Retrying... ($RETRIES/$MAX_RETRIES)"
done
echo "FastAPI service is available."

# =============================================================================
# Run training
# =============================================================================

# Activate the virtual environment and run training
source $HOME/{{ mixtape }}/bin/activate
# Run the training script. Pass in the port where the server is running.
python -u run_training.py -p $PORT
