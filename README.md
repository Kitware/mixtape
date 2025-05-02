# MIXTAPE

## Develop with Docker (recommended quickstart)
This is the simplest configuration for developers to start with.

### Initial Setup
1. Run `docker compose run --rm django ./manage.py migrate`
2. Run `docker compose run --rm django ./manage.py createsuperuser`
   and follow the prompts to create your own user

### Run Application
1. Run `docker compose up`
2. Access the site, starting at http://localhost:8000/admin/
3. When finished, use `Ctrl+C`

### Application Maintenance
Occasionally, new package dependencies or schema changes will necessitate
maintenance. To non-destructively update your development stack at any time:
1. Run `docker compose pull`
2. Run `docker compose build --pull --no-cache`
3. Run `docker compose run --rm django ./manage.py migrate`

## Develop Natively (advanced)
This configuration still uses Docker to run attached services in the background,
but allows developers to run Python code on their native system.

### Initial Setup
1. Run `docker compose -f ./docker-compose.yml up -d`
2. Install Python 3.11
3. Create and activate a new Python virtualenv
4. Run `pip install -e .[dev]`
5. Run `source ./dev/export-env.sh`
6. Run `./manage.py migrate`
7. Run `./manage.py createsuperuser` and follow the prompts to create your own user

### Run Application
1.  Ensure `docker compose -f ./docker-compose.yml up -d` is still active
2. Run:
   1. `source ./dev/export-env.sh`
   2. `./manage.py runserver`
3. Run in a separate terminal:
   1. `source ./dev/export-env.sh`
   2. `celery --app mixtape.celery worker --loglevel INFO --without-heartbeat`
4. When finished, run `docker compose stop`
5. To destroy the stack and start fresh, run `docker compose down -v`

## Add data

### Training
Start by training an environment of your choice:
```bash
Usage: manage.py training [OPTIONS]

  Run training on the specified environment.

Options:
  --version                       Show the version and exit.
  -h, --help                      Show this message and exit.
  -e, --env_name [knights_archers_zombies_v10|pistonball_v6|cooperative_pong_v5|BattleZone-v5|Berzerk-v5|ChopperCommand-v5]
                                  The PettingZoo or Gymnasium environment to
                                  use.
  -a, --algorithm [PPO|DQN]       The RLlib algorithm to use.
  -p, --parallel                  All agents have simultaneous actions and
                                  observations.
  -g, --num_gpus FLOAT            Number of GPUs to use.
  -t, --training_iteration INTEGER
                                  Number of training iterations to run.
  -f, --config_file FILENAME      Arguments to configure the environment.
  --immediate                     Run the task immediately.
```

For example:
```bash
# This will use all of the default values (with the exception of parallel):
# environment: 'knights_archers_zombies_v10'
# algorithm: 'PPO'
# parallel: True
# training_iteration: 100
# config_file: None
python manage.py training -p
```

### Inference

Select an existing checkpoint to run inference:
```bash
Usage: manage.py inference [OPTIONS] CHECKPOINT_PK

  Run inference on the specified trained environment.

Options:
  --version                       Show the version and exit.
  -h, --help                      Show this message and exit.
  -e, --env_name [knights_archers_zombies_v10|pistonball_v6|cooperative_pong_v5|BattleZone-v5|Berzerk-v5|ChopperCommand-v5]
                                  The PettingZoo or Gymnasium environment to
                                  use.
  -f, --config_file FILENAME      Arguments to configure the environment.
  -p, --parallel                  All agents have simultaneous actions and
                                  observations.
  --immediate                     Run the task immediately.
```

For example:
```bash
# This will use the checkpoint with ID 1 and will use the
# parallel environment if it is a PettingZoo environment
python manage.py inference 1 -p
```

To see all available checkpoints visit http://localhost:8000/admin/core/checkpoint/.

### Ingest existing episode(s)

See the [ingest documentation](./mixtape/core/ingest/README.md)

## Testing
### Initial Setup
tox is used to execute all tests.
tox is installed automatically with the `dev` package extra.

When running the "Develop with Docker" configuration, all tox commands must be run as
`docker compose run --rm django tox`; extra arguments may also be appended to this form.

### Running Tests
Run `tox` to launch the full test suite.

Individual test environments may be selectively run.
This also allows additional options to be be added.
Useful sub-commands include:
* `tox -e lint`: Run only the style checks
* `tox -e type`: Run only the type checks
* `tox -e test`: Run only the pytest-driven tests

To automatically reformat all code to comply with
some (but not all) of the style checks, run `tox -e format`.
