import os

import django
from django.urls import resolve

os.environ.setdefault("UV_ENV_FILE", "./dev/.env.docker-compose")
django.setup()
resolve("/")
