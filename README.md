# MIXTAPE

## Develop with Docker
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

### Training - Supported environments

- **Knights Archers Zombies** (`knights_archers_zombies_v10`, PettingZoo)
  - Multi‑agent, discrete actions. Demonstrates combat/strategy coordination.
  - Example:
    ```bash
    docker compose run --rm django \
    ./manage.py training -e knights_archers_zombies_v10 \  # Environment
    -a PPO \                                               # Agent
    -p \                                                   # Parallel
    -g 0.0 \                                               # GPUs
    -t 100 \                                               # Iterations
    --immediate
    ```

- **Pistonball** (`pistonball_v6`, PettingZoo)
  - Multi‑agent with a continuous action space. Demonstrates continuous control and teamwork.
  - Example:
    ```bash
    docker compose run --rm django \
    ./manage.py training -e pistonball_v6 \  # Environment
    -a PPO \                                 # Agent
    -g 0.0 \                                 # GPUs
    -t 100 \                                 # Iterations
    --immediate
    ```

- **LunarLander** (`LunarLander-v2`, Gymnasium)
  - Single‑agent, discrete actions. Demonstrates balancing multiple variables to achieve a safe, stable landing. Ideal candidate for decomposed rewards.
  - Example:
    ```bash
    docker compose run --rm django \
    ./manage.py training -e LunarLander-v2 \  # Environment
    -a PPO \                                  # Agent
    -g 0.0 \                                  # GPUs
    -t 100 \                                  # Iterations
    --immediate
    ```

Notes:
- Use `-p/--parallel` only for PettingZoo environments.
- DQN is for discrete action spaces; it is not available for Pistonball (continuous).

### Inference

Review available checkpoints:
```bash
docker compose run --rm django ./manage.py list_checkpoints
```

You will see a list of available checkpoints, with the most recent at the top.
```bash
environment                 | checkpoint_pk | created             | inferences | episodes
----------------------------+---------------+---------------------+------------+---------
pistonball_v6               | 2             | 2026-01-03 19:29:25 | 1          | 1
knights_archers_zombies_v10 | 1             | 2026-01-03 19:27:32 | 1          | 1
```

Select an existing checkpoint to run inference. For example:
```bash
docker compose run --rm django ./manage.py inference 2 -p --immediate
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
