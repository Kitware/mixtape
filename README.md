# MIXTAPE

## Description
This project aims to deliver middleware and visualization tools that facilitate the interactive explanation and visualization of AI decision-making systems.

MIXTAPE utilizes Ray and Ray RLLib, along with the standardized frameworks Gymnasium and PettingZoo, to create a scalable platform. This architecture supports both pre-configured environments and agents, as well as the flexibility for users to integrate their own custom environments and agents. Through simple API endpoints, accessible via the command line or Swagger interactive documentation, users can easily train models, run inference tasks, and generate explanations. These results, along with summary information, can be seamlessly visualized through the web UI.

## Getting Started
Start the Docker container:
```bash
cd {path-to-mixtapeii-repo}/mixtapeii
docker compose up
```

## Usage / Examples

Navigate to http://localhost:5000/docs to use the interactive docs.

### Training
```
train(
    env_to_register: ButterflyEnvs,
    env_config: Dict,
    parallel: bool = True,
    num_gpus: int = 0,
    training_iteration: int = 100,
    env_args: Dict = None,
    rollout_args: Dict = None,
    training_args: Dict = None,
    framework_args: Dict = None,
    run_args: Dict = None
) -> None
```
Run training and produce logs as well as a gif for each episode. The logs produced are a json file containing the observation spaces, actions, and rewards for each actor per step, as well as the total reward for that episode.

**Parameters:**

- **env_to_register**: Select from one of the available PettingZoo [butterfly](https://pettingzoo.farama.org/environments/butterfly/) environments.
- **env_config**: A dict of arguments to be passed in the environment creation step (see docs for [cooperative pong](https://pettingzoo.farama.org/environments/butterfly/cooperative_pong/#arguments), [knights archers zombies](https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/#arguments), and [pistonball](https://pettingzoo.farama.org/environments/butterfly/pistonball/#arguments) arguments).
- **parallel**: Whether or not to use the parallel environment where all agents have simultaneous actions and observations.
- **num_gpus**: Number of GPUs to allocate to the algorithm process. Note that not all algorithms can take advantage of GPUs. This can be fractional (e.g., 0.3 GPUs).
- **training_iteration**: Stop trials after reaching a training_iteration number of iterations
- **env_args**: Sets the config’s RL-environment settings ([See docs for details](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.environment.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig.environment))
- **training_args**: Sets the training related configuration ([See docs for details](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.training.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig.training))
- **framework_args**: Sets the config’s DL framework settings([See docs for details](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.framework.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig.framework))
- **run_args**: Runtime configuration for training and tuning runs ([See docs for details](https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html#ray.train.RunConfig))

### Inference
```
inference(
    env_to_register: ButterflyEnvs,
    env_config: Dict,
    checkpoint_path: str,
    parallel: bool = False
) -> None
```

Run inference and produce logs as well as a gif for each episode. The logs produced are a json file containing the observation spaces, actions, and rewards for each actor per step, as well as the total reward for that episode.

**Parameters:**

- **env_to_register**: Select from one of the available PettingZoo [butterfly](https://pettingzoo.farama.org/environments/butterfly/) environments.
- **env_config**: A dict of arguments to be passed in the environment creation step (see docs for [cooperative pong](https://pettingzoo.farama.org/environments/butterfly/cooperative_pong/#arguments), [knights archers zombies](https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/#arguments), and [pistonball](https://pettingzoo.farama.org/environments/butterfly/pistonball/#arguments) arguments).
- **checkpoint_path**: The path (str) to a Policy or Algorithm checkpoint directory instance to restore from.
- **parallel**: Whether or not to use the parallel environment where all agents have simultaneous actions and observations.

## Cleanup

Stop the service and remove unneeded temporary docker volumes:
```bash
docker compose down -v
```

## Contributing

Clone the repo

```bash
git clone https://gitlab.kitware.com/mixtape/mixtapeii.git
cd mixtapeii
```

It's strongly recommended that you use a virtual environment of your choice.
- **venv** [[user guide](https://docs.python.org/3/library/venv.html#creating-virtual-environments)]
- **micromamba** [[installation](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)] [[user guide](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)]
- **conda** [[installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)] [[user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#getting-started-with-conda)]

Install dependencies

``` bash
pip install -r requirements.txt
pre-commit install
```

From now on `pre-commit` will automatically run `Black` on files that you've modified. If formatting changes are needed, `pre-commit` will update files before committing.
