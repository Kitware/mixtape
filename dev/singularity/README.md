# Singularity Deployment

This directory contains scripts that will build a singularity deployment of the resonant application. Root access (or `sudo`) is required.

## Steps

First, ensure you're in the right directory

```
cd dev/singularity
```

Then, you must build the singularity images.

```
./build-images.sh
```

Note, if you've already run this before, it will prompt you to confirm or reject replacing the existing image. You can avoid this by first running `rm ./images/*`, or by specifically deleting the `.sif` file you want to rebuild.

Now, you're ready to run the singularity containers. You can do this by simply running:

```
./run-containers.sh
```

This will run these containers in the background. You can see what is running with `singularity instance list`. Note that singularity has no internal network like `docker compose` does, and so any ports used by the application will be used on the host machine (and as such, is subject to conflict with other services you may have running locally).

Once you're done, you can stop the running instances with

```
./stop-containers.sh
```
