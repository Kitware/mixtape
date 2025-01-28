release: ./manage.py migrate
web: gunicorn --bind 0.0.0.0:$PORT mixtape.wsgi
worker: REMAP_SIGTERM=SIGQUIT celery --app mixtape.celery worker --loglevel INFO --without-heartbeat
