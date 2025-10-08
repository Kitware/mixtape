import os

# This is only used when building a Docker image, so "collectstatic" can run
# without externally setting environment variables.

os.environ['DJANGO_DATABASE_URL'] = 'postgres://postgres:postgres@localhost:5432/django'
os.environ['DJANGO_CELERY_BROKER_URL'] = 'amqp://localhost:5672/'
os.environ['DJANGO_MINIO_STORAGE_URL'] = (
    'http://minioAccessKey:minioSecretKey@localhost:9000/django-storage'
)
os.environ['DJANGO_ALLOWED_HOSTS'] = 'localhost'
os.environ['DJANGO_SECRET_KEY'] = 'insecure-secret'

from .production import *  # isort: skip
