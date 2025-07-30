# MIXTAPE

## Develop with Docker (recommended quickstart)
This is the simplest configuration for developers to start with.

### Initial Setup
1. Run `docker compose run --rm django ./manage.py migrate`
2. Run `docker compose run --rm django ./manage.py createsuperuser`
   and follow the prompts to create your own user

### Run Application
1. Run `docker compose up`
2. Access the site, starting at <http://localhost:8000/admin/>
3. When finished, use `Ctrl+C`

### Maintenance
To non-destructively update your development stack at any time:
1. Run `docker compose down`
2. Run `docker compose pull`
3. Run `docker compose build --pull`
4. Run `docker compose run --rm django ./manage.py migrate`

### Destruction
1. Run `docker compose down -v`

## Add data

### Training
Start by training an environment of your choice. For example:
```bash
docker compose run --rm django \
./manage.py training \
-e knights_archers_zombies_v10 \
-a PPO \
-p \
-g 0.0 \
-t 100 \
--immediate

# For a detailed breakdown of all available options, use -h|--help
docker compose run --rm django ./manage.py training --help
```

### Inference

Select an existing checkpoint to run inference. For example:
```bash
docker compose run --rm django \
./manage.py inference \
1 \
-p \
--immediate

# For a detailed breakdown of all available options, use -h|--help
docker compose run --rm django ./manage.py inference --help
```

If you've already started the server with `docker compose up`, you can see all available checkpoints at <http://localhost:8000/admin/core/checkpoint/>.

### Ingest existing episode(s)

See the [ingest documentation](./mixtape/core/ingest/README.md)

## Testing
### Initial Setup
When running the "Develop with Docker" configuration, all tox commands must be run as
`docker compose run --rm django uv run tox`; extra arguments may also be appended to this form.

### Running Tests
Run `uv run tox` to launch the full test suite.

Individual test environments may be selectively run.
This also allows additional options to be be added.
Useful sub-commands include:
* `uv run tox -e lint`: Run only the style checks
* `uv run tox -e type`: Run only the type checks
* `uv run tox -e test`: Run only the pytest-driven tests

To automatically reformat all code to comply with
some (but not all) of the style checks, run `uv run tox -e format`.
