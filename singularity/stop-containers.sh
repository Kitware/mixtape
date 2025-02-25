#!/usr/bin/env bash

singularity instance stop django
singularity instance stop celery
singularity instance stop postgres
singularity instance stop rabbitmq
singularity instance stop minio
