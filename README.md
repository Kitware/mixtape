# MIXTAPE

## Description
Explainable AI (XAI) middleware and visualation tools that support the interactive explanation and visualization og AI decision-making systems.

## Getting Started
Start the Docker container to make the API available:
```bash
cd {path-to-mixtapeii-repo}/mixtapeii
docker build -t mixtape-fastapi -f devops/fastapi/Dockerfile .
docker compose -f devops/docker-compose.fastapi.yml up
```

## Usage / Examples

Navigate to http://localhost:5000/docs to use the interactive docs.

### Training

**train(env_to_register: ButterflyEnvs, env_config: Dict, parallel: bool = True, num_gpus: int = 0, timesteps_total: int = 100, env_args: Dict = None, rollout_args: Dict = None, training_args: Dict = None, framework_args: Dict = None, run_args: Dict = None) -> None**

Run training and produce logs as well as a gif for each episode. The logs produced are a json file containing the observation spaces, actions, and rewards for each actor per step, as well as the total reward for that episode.

**Parameters:**

- **env_to_register**: Select from one of the available PettingZoo [butterfly](https://pettingzoo.farama.org/environments/butterfly/) environments.
- **env_config**: A dict of arguments to be passed in the environment creation step (see docs for [cooperative pong](https://pettingzoo.farama.org/environments/butterfly/cooperative_pong/#arguments), [knights archers zombies](https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/#arguments), and [pistonball](https://pettingzoo.farama.org/environments/butterfly/pistonball/#arguments) arguments).
- **parallel**: Whether or not to use the parallel environment where all agents have simultaneous actions and observations.
- **num_gpus**: Number of GPUs to allocate to the algorithm process. Note that not all algorithms can take advantage of GPUs. This can be fractional (e.g., 0.3 GPUs).
- **timesteps_total**: Total number of timesteps
- **env_args**: Sets the config’s RL-environment settings ([See docs for details](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.environment.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig.environment))
- **training_args**: Sets the training related configuration ([See docs for details](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.training.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig.training))
- **framework_args**: Sets the config’s DL framework settings([See docs for details](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.framework.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig.framework))
- **run_args**: Runtime configuration for training and tuning runs ([See docs for details](https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html#ray.train.RunConfig))

### Inference

**inference(env_to_register: ButterflyEnvs, env_config: Dict, checkpoint_path: str, parallel: bool = False)**

Run inference and produce logs as well as a gif for each episode. The logs produced are a json file containing the observation spaces, actions, and rewards for each actor per step, as well as the total reward for that episode.

**Parameters:**

- **env_to_register**: Select from one of the available PettingZoo [butterfly](https://pettingzoo.farama.org/environments/butterfly/) environments.
- **env_config**: A dict of arguments to be passed in the environment creation step (see docs for [cooperative pong](https://pettingzoo.farama.org/environments/butterfly/cooperative_pong/#arguments), [knights archers zombies](https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/#arguments), and [pistonball](https://pettingzoo.farama.org/environments/butterfly/pistonball/#arguments) arguments).
- **checkpoint_path**: The path (str) to a Policy or Algorithm checkpoint directory instance to restore from.
- **parallel**: Whether or not to use the parallel environment where all agents have simultaneous actions and observations.

## Local Development / Testing

Running locally is recommended for faster development.

```bash
pip install fastapi/requirements.txt
cd fastapi/
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

The `reload` flag automatically reloads the server when you make changes. The interactive docs will be available at http://localhost:5000/docs
