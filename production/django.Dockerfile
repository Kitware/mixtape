FROM ghcr.io/astral-sh/uv:debian
# Make Python more friendly to running in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

# Enable caching of Python versions
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python \
# Don't expect the package cache to be available at runtime
  UV_LINK_MODE=copy \
# Production install options
  UV_LOCKED=true \
  UV_NO_DEV=true

WORKDIR /opt/django-project

# Install only dependencies on a separate layer, to improve cache hits
RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=.python-version,target=.python-version \
  uv sync --compile-bytecode --no-install-project

COPY ["./pyproject.toml", "./uv.lock", "./.python-version", "./manage.py", "./"]
COPY ["./mixtape", "./mixtape"]

# Install the actual project (dependencies should already be available)
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --compile-bytecode

RUN DJANGO_SETTINGS_MODULE=mixtape.settings.build ./manage.py collectstatic --noinput

# TODO: Consider baking DJANGO_SENTRY_RELEASE into this, based on the Git commit
