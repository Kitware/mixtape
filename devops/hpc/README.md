# Running on HPC Resources

This provides a simple example of how you can use Ray and run a simple script on HPC resources.

## Conda Setup

This example uses a conda environment. This can be setup as follows:

1. Load the module
    ```bash
    module load cse/anaconda3/latest
    ```

2. Create an environment that uses conda-forge
    ```bash
    conda create -n mixtape python=3.10
    conda activate mixtape
    ```

3. Configure Conda to use the conda-forge channel
    ```bash
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    ```

4. Install the required packages
    ```bash
    conda install \
        ray-default=2.40.0 \
        ray-rllib=2.40.0 \
        pytorch=2.1.0 \
        pillow=11.0.0 \
        numpy=1.24.0 \
        pettingzoo=1.24.3
    ```
