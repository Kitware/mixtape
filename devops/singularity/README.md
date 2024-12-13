# Singularity

Singularity is a container framework specifically designed for running scientific applications on high-performance computing (HPC) systems. It addresses common challenges in HPC environments by prioritizing security, streamlining deployment, and enabling seamless integration with local resources. Additionally, it supports advanced workflows, such as those leveraging MPI and GPU acceleration, making it an ideal solution for advanced workflows like those that MIXTAPE supports.

## How to create

### Install Singularity

The container technology known as "Singularity" was (relatively) recently split into two similar projects: [Apptainer](https://apptainer.org/), an open-source branch maintained by the Linux Foundation, and [Singularity](https://sylabs.io/), which continues to be developed by Sylabs.

While Apptainer is the supported version on most DSRC platforms, the core functionality of both remains largely the same, and containers created for Singularity typically run seamlessly on Apptainer. However, as current training materials focus on Singularity and the Narwhal system does not yet support Apptainer, these instructions will specifically reference Singularity.

- [Install Apptainer](https://apptainer.org/docs/admin/1.3/installation.html#)
- [Install Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html)

### Create a SIF file

A `SIF` file is a compressed `SquashFS` filesystem in the "Singularity Image Format". MIXTAPE uses its existing Docker build process to create the necessary Singularity `SIF` file.

Make sure your build is up-to-date and export Docker image to tar file.

```bash
cd devops/docker

docker build -t mixtape ./fastapi
docker save -o mixtape.tar mixtape
```

Convert Docker tar to Singularity sif

```bash
singularity build mixtape.sif docker-archive://mixtape.tar
```

Move sif file and example bash and Python scripts to the HPC resource

```bash
scp mixtape.sif bash_script.sh run_training.py {user_name}@narwhal.navydsrc.hpc.mil:/p/home/{user_name}
```

## How to use

### HPC

#### Script Directives

At the top of the example `bash_script.sh` file you will see a series of PBS directives: `#PBS -{flag} {args}`. These define resource requests and other scheduler directives. The example file directives are as follows:

- Specify the shell that should be used to interpret the job script

    ```bash
    #PBS -S /bin/bash
    ```

- How long the job should run (maximum time - job will end sooner if it completes sooner)
    ```bash
    #PBS -l walltime=00:10:00
    ```
- ID of project to charge (use `show_usage` to list projects available)
    ```bash
    #PBS -A 0000000000000
    ```

- Redirect stderr and stdout into stdout and write to file `mixtape.out`
    ```bash
    #PBS -o mixtape.out
    #PBS -j oe
    ```

- Number of nodes, cores, and processes
    ```bash
    #PBS -l select=1:ncpus=128:mpiprocs=2:ngpus=1
    ```

- The queue to use
    ```bash
    #PBS -q debug
    ```

- Job directive name
    ```bash
    #PBS -N simple_test
    ```

- Send an email on job end to `email_address@domain.com`
    ```bash
    #PBS -m e
    #PBS -M email_address@domain.com
    ```

See the [PBS Guide](https://centers.hpc.mil/users/docs/navy/narwhalPbsGuide.html) for more detailed information and additional directives that may be of use.

#### Script Updates

Before submitting a job there are a few pieces of the script that will need to be updated.

1. `#PBS -A {{ PROJECT ID }}`
    - Replace `{{ PROJECT ID }}` with the id of the project that should be charged.
2. `#PBS -N {{ JOB NAME }}`
    - Replace `{{ JOB NAME }}` with an easily identifiable name so that you can locate output easily.
3. `#PBS -M {{ EMAIL ADDRESS }}`
    - `{{ EMAIL ADDRESS }}` should be replaced with your email address. Optionally, you can also remove this line as well as `#PBS -m e` to recieve no emails at all.
4. `PORT="{{ 8000 }}"`
    - Remove the curly braces around `8000` and either replace it with a different port to use or use the default port.
5. `{{ cd ${PBS_O_WORKDIR} }}`
    - Update this `cd` command to change to the directory that contains the `SIF` file.
6. `--bind {{  ${PBS_O_WORKDIR}/logs:/app/logs  }} \`
    - If the logs should be written to a different location, update that here.
7. `WAIT_TIME={{ 5 }}`
    - The number of seconds to wait before retrying the server to see if it is available. Remove the curly braces and update if desired.
8. `MAX_RETRIES={{ 24 }}`
     - The maximum number of retries to attempt before giving up on the server. Remove the curly braces and update if desired.
9. `source $HOME/{{ mixtape }}/bin/activate`
    - In the next step you will create a virtual environment. If you decide to name it something other than `mixtape` update that here.

Before running confirm that all double curly braces have been removed.

#### Running on HPC Resources

Follow the [instructions](https://centers.hpc.mil/users/index.html) for information on how to install Kerberos for your system, how to initialize a ticket, and how to connect to the system via `ssh`.

If you copied the files to your home directory (as shown in the example above) you should see the files now when you run `ls`. Otherwise you will need to naviagte to the location you copied your files to.

Before submitting the script you will need to set up a virtual environment so that you can install the `httpx` library. This is what the `run_training.py` script uses to call the training endpoint.

```bash
pip install virtualenv             # Install virtualenv
python -m venv $HOME/mixtape       # Create a virtual environment named "mixtape"
source $HOME/mixtape/bin/activate  # Activate the environment
pip install httpx                  # Install httpx
```

If you have any additional dependencies they should also be installed in this environment so they are available when the job is run.

You can now submit submit the bash script to run

```bash
qsub bash_script.sh
```

Check on the status of your queued and running jobs with `qstat`

```bash
qstat -u bnmajor
```

Some of the most common commands you may need:

| Command | Description|
|---------|------------|
|cqstat | Display running and pending jobs, including estimated start times
|qdel job_id | Delete a job
|qstat job_id | Check the status of a job
|qstat -q | Display the status of all queues
|qsub script_file | Submit a job

You can use `man {command}` or `man {command} --help` for more information about commands. You can also see the [documentation](https://centers.hpc.mil/users/docs/navy/narwhalPbsGuide.html#submitManage) for additional commands that may be of use.

### Locally

To run the container locally you can run

```bash
singularity exec \
    --bind {path-to-logs}/logs:/app/logs \
    mixtape.sif \
    bash -c "PYTHONPATH=/app uvicorn app.main:app --host 0.0.0.0 --port 8000"
```

You can then navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to use the interactive Swagger docs to test endpoints. You can also run the `run_training.py` example Python script.
