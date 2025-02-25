#!/usr/bin/env bash

# Normalize script path, so it's the same regardless of where it's called from
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEFINITIONS_DIR=$SCRIPT_DIR/definitions
IMAGES_DIR=$SCRIPT_DIR/images
mkdir -p $DEFINITIONS_DIR
mkdir -p $IMAGES_DIR

# Build dependent images
sudo singularity build $IMAGES_DIR/postgres.sif  $DEFINITIONS_DIR/postgres.def
sudo singularity build $IMAGES_DIR/rabbitmq.sif  $DEFINITIONS_DIR/rabbitmq.def
sudo singularity build $IMAGES_DIR/minio.sif     $DEFINITIONS_DIR/minio.def

# Build django and celery images
sudo singularity build -F $IMAGES_DIR/django.sif $DEFINITIONS_DIR/django.def
sudo singularity build -F $IMAGES_DIR/celery.sif $DEFINITIONS_DIR/celery.def
