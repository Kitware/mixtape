# Singularity Deployment

This directory contains scripts that will build a singularity deployment of the resonant application. Root access (or `sudo`) is required.

## Steps

### Building Images

First, ensure you're in the right directory

```
cd singularity
```

Then, you must build the singularity images.

```
./build-images.sh
```

Note, if you've already run this before, it will prompt you to confirm or reject replacing the existing image. You can avoid this by first running `rm ./images/*`, or by specifically deleting the `.sif` file you want to rebuild.

### Env vars

The following env vars can be set in the `.env.singularity` file if desired, but have (insecure) defaults. If there is no connection to the external network, this is not a concern.

- POSTGRES_DB - The name of the postgres database that django will use. Defaults to `django`.
- POSTGRES_PASSWORD - The password for the `postgres` superuser. Defaults to `postgres`.
- MINIO_ROOT_USER - The root minio user. Defaults to `minioAccessKey`.
- MINIO_ROOT_PASSWORD - The password for the root minio user. Defaults to `minioSecretKey`.

### Running the containers

Now, you're ready to run the singularity containers. You can do this by simply running:

```
./run-containers.sh
```

This will run these containers in the background. You can see what is running with `singularity instance list`. Note that singularity has no internal network like `docker compose` does, and so any ports used by the application will be used on the host machine (and as such, is subject to conflict with other services you may have running locally).

### Running commands

To run simple commands, like creating the first admin user in the system, you can run a command like so (ran from the project root):

```
singularity exec \
    --env-file singularity/.env.singularity \
    instance://django \
    ./manage.py createsuperuser
```

The command `./manage.py createsuperuser` is used as an example, but that may be substituted with any available command.

### Stopping the containers

Once you're done, you can stop the running instances with

```
./stop-containers.sh
```
