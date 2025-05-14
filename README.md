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

## Add data

### Training
Start by training an environment of your choice. For example:
```bash
# This will use all of the default values (with the exception of parallel):
# environment: 'knights_archers_zombies_v10'
# algorithm: 'PPO'
# parallel: True
# num_gpus: 0.0
# training_iteration: 100
# config_file: None
python manage.py training -p

# This can also be specified explicity
python manage.py training -e knights_archers_zombies_v10 -a PPO -p -g 0.0 -t 100

# For a detailed breakdown of all available options, use -h|--help
python manage.py training --help
```

### Inference

Select an existing checkpoint to run inference. For example:
```bash
# This will use the checkpoint with ID 1 and will use the
# parallel environment if it is a PettingZoo environment
python manage.py inference 1 -p

# This can also be specified explicity
python manage.py inference -e knights_archers_zombies_v10 -p

# For a detailed breakdown of all available options, use -h|--help
python manage.py inference --help
```

If you've already started the server with `docker compose up`, you can see all available checkpoints at http://localhost:8000/admin/core/checkpoint/.

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
