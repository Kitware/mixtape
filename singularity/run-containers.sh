#!/usr/bin/env bash

# Normalize script path, so it's the same regardless of where it's called from
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ENV_FILE=$SCRIPT_DIR/.env.singularity
PROJECT_DIR=$SCRIPT_DIR/..

# Ensure bind mount directories exist
VOLUME_DIR=$SCRIPT_DIR/volumes
mkdir -p $VOLUME_DIR

POSTGRES_VOLUME=$VOLUME_DIR/postgres
mkdir -p $POSTGRES_VOLUME/data
mkdir -p $POSTGRES_VOLUME/run

RABBITMQ_VOLUME=$VOLUME_DIR/rabbitmq
mkdir -p $RABBITMQ_VOLUME

MINIO_VOLUME=$VOLUME_DIR/minio
mkdir -p $MINIO_VOLUME


# Run postgres
singularity instance start \
  --bind $POSTGRES_VOLUME/data:/var/lib/postgresql/data:rw \
  --bind $POSTGRES_VOLUME/run:/var/run/postgresql:rw \
  --env-file=$ENV_FILE \
  images/postgres.sif postgres

# Run RabbitMQ
singularity instance start --bind $RABBITMQ_VOLUME:/var/lib/rabbitmq images/rabbitmq.sif rabbitmq

# Run MinIO
singularity instance start \
  --env-file $ENV_FILE \
  --bind $MINIO_VOLUME:/data \
  images/minio.sif minio

# Run Django + Celery
# Run django first so that migrations are applied
singularity instance start \
  --bind $PROJECT_DIR:/opt/django-project \
  --env-file $ENV_FILE \
  images/django.sif django

# Run celery
singularity instance start \
  --bind $PROJECT_DIR:/opt/django-project \
  --env-file $ENV_FILE \
  images/celery.sif celery
