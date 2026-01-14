import logging

import sentry_sdk
import sentry_sdk.integrations.celery
import sentry_sdk.integrations.django
import sentry_sdk.integrations.logging
import sentry_sdk.integrations.pure_eval

from .base import *

# Import these afterwards, to override
# from resonant_settings.production.email import *  # isort: skip
from resonant_settings.production.https import *  # isort: skip
from resonant_settings.development.minio_storage import *  # isort: skip

WSGI_APPLICATION = 'mixtape.wsgi.application'

SECRET_KEY: str = env.str('DJANGO_SECRET_KEY')

STORAGES['default'] = {
    'BACKEND': 'minio_storage.storage.MinioMediaStorage',
}

# This only needs to be defined in production. Testing will add 'testserver'. In development
# (specifically when DEBUG is True), 'localhost' and '127.0.0.1' will be added.
ALLOWED_HOSTS: list[str] = env.list('DJANGO_ALLOWED_HOSTS', cast=str)

# Assume we are always behind a proxy setting "X-Forwarded-Proto" and "X-Forwarded-Host"
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
USE_X_FORWARDED_HOST = True

_proxy_subpath: str | None = env.str('DJANGO_MIXTAPE_PROXY_SUBPATH', default=None)
if _proxy_subpath:
    FORCE_SCRIPT_NAME = _proxy_subpath
    # Work around https://code.djangoproject.com/ticket/36653
    STORAGES['staticfiles'].setdefault('OPTIONS', {})['base_url'] = f'{_proxy_subpath}/{STATIC_URL}'

# sentry_sdk is able to directly use environment variables like 'SENTRY_DSN', but prefix them
# with 'DJANGO_' to avoid conflicts with other Sentry-using services.
sentry_sdk.init(
    dsn=env.str('DJANGO_SENTRY_DSN', default=None),
    environment=env.str('DJANGO_SENTRY_ENVIRONMENT', default=None),
    release=env.str('DJANGO_SENTRY_RELEASE', default=None),
    integrations=[
        sentry_sdk.integrations.logging.LoggingIntegration(
            level=logging.INFO,
            event_level=logging.WARNING,
        ),
        sentry_sdk.integrations.django.DjangoIntegration(),
        sentry_sdk.integrations.celery.CeleryIntegration(),
        sentry_sdk.integrations.pure_eval.PureEvalIntegration(),
    ],
    # "project_root" defaults to the CWD, but for safety, don't assume that will be set correctly
    project_root=str(BASE_DIR),
    # Send traces for non-exception events too
    attach_stacktrace=True,
    # Submit request User info from Django
    send_default_pii=True,
)
