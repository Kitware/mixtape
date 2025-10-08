FROM ghcr.io/astral-sh/uv:debian
# Make Python more friendly to running in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

WORKDIR /opt/django-project
COPY ./mixtape ./mixtape/
COPY ./pyproject.toml ./uv.lock ./manage.py ./.python-version ./
RUN uv sync --no-dev --locked --no-cache

RUN DJANGO_SETTINGS_MODULE=mixtape.settings.build ./manage.py collectstatic --noinput

# TODO: Consider baking DJANGO_SENTRY_RELEASE into this, based on the Git commit
