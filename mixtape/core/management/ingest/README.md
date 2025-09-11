# External Episode Ingestion

This directory contains functionality for ingesting external episodes into the Mixtape system. This document explains how to create a valid JSON file and how to ingest it.

## JSON File Structure

The JSON file is required to contain information about both the training as well as inference. The following is a breakdown of the required format:

```jsonc
{
    /**
    action_mapping: Optional.
        If provided, this will map integer action values to string values.
        This is used in the UI for improved labeling of results.
        Top level key-pairs are agent actions to string values.
        Nested key-pairs are unit actions to string values.
        Continuous action spaces should skip this.
    */
    "action_mapping": {
        "0": "string",
        "1": "string",
        "unit_mapping": {
            "0": "string",
            "1": "string"
        }
    },
    "training": {
        "environment": "string",
        "algorithm": "string",
        "parallel": false,                          // Optional, defaults to false
        "num_gpus": 0.0,                            // Optional
        "iterations": 100,
        "config": {},                               // Optional. Any valid JSON object with config settings for the training environment.
        "reward_mapping": ["string"]                // Optional. If provided, this will map integer reward values to string values.
    },
    "inference": {
        "parallel": false,                          // Optional, defaults to false
        "config": {},                               // Optional. Any valid JSON object with config settings for the inference environment.
        "steps": [                                  // Must have at least one step
            {
                "number": 0,                        // This is the step number
                "image": "base64_encoded_string",   // Optional
                "agent_steps": [                    // Optional, one agent_step per agent for the current step
                    {
                        "agent": "string",
                        /**
                        action: required
                            This is the action value that will be mapped to the corresponding
                            string if "action_mapping" is provided. Continuous action spaces should
                            provide float values, discrete action spaces should provide integer
                            values.
                        */
                        "action": "float | int",
                        "reward": "float | int",              // Users must provide either reward or rewards, not both
                        "rewards": ["float | int"],           // Users must provide either reward or rewards, not both
                        "observation_space": ["float | int"], // Currently only supports 1D or 2D array of floats
                        "action_distribution": ["float"],   // Optional
                        /**
                        health: Optional.
                            Object containing health metrics for agents or units.
                            For example: {"friendly": 100.0, "enemy": 75.5}
                        */
                        "health": "float" | "int",
                        /**
                        value_estimate: Optional.
                            The predicted value estimate from the agent's value function.
                        */
                        "value_estimate": "float",
                        /**
                        predicted_reward: Optional.
                            The predicted reward from the agent's policy.
                        */
                        "predicted_reward": "float",
                        /**
                        enemy_agent_health: Optional.
                            Object containing health metrics for enemy agents.
                            For example: [100.0, 75.5]
                        */
                        "enemy_agent_health": ["float" | "int"],
                        /**
                        enemy_unit_health: Optional.
                            Object containing health metrics for enemy units.
                            For example: [100.0, 75.5]
                        */
                        "enemy_unit_health": ["float" | "int"],
                        /**
                        custom_metrics: Optional.
                            Any custom metrics that you would like to store for the agent step.
                            Each top-level key is a plot title that points to an object.
                            Each nested object key is an axis ("x", "y", "z", etc.) or an axis title
                            ("x_label", "y_label", etc.) that points to an array of values.
                            Must be valid JSON.
                        */
                        "custom_metrics": {},
                        /**
                        unit_steps: Optional.
                            An array of unit steps for team-based agents where individual units can have their own actions and metrics.
                            Each unit step can have the same fields as agent_step (action, rewards, health, etc.).
                            The agent_step represents the team-level action, while unit_steps represent individual unit actions.
                        */
                        "unit_steps": [
                            {
                                "unit": "string",
                                "action": "float | int",
                                "reward": "float | int",              // Users must provide either reward or rewards, not both
                                "rewards": ["float | int"],           // Users must provide either reward or rewards, not both
                                "health": "float | int",              // Optional. Value representing health metrics for units.
                                "value_estimate": "float",
                                "predicted_reward": "float",
                                /**
                                custom_metrics: Optional.
                                    Any custom metrics that you would like to store for the unit step.
                                    Each top-level key is a plot title that points to an object.
                                    Each nested object key is an axis ("x", "y", "z", etc.) or an axis title
                                    ("x_label", "y_label", etc.) that points to an array of values.
                                    Must be valid JSON.
                                */
                                "custom_metrics": {}
                            }
                        ]
                    }
                ]
            }
        ]
    }
}
```

## Ingesting the JSON File

To ingest an external episode, use the Django management command:

```bash
docker compose run --rm django ./manage.py ingest_episode path/to/your/episode.json
```

The command will validate the JSON structure against the required schema and create all the required records.

## Errors

The ingest process will fail if:
- The JSON file is not formatted correctly
- Required values are missing
- Values don't meet validation requirements
- The base64-encoded image data (if provided) is invalid

If any part of the ingestion fails, no partial data will be saved to the database.

## Example

The simple [Stable Baselines example](./simple_baselines_lunar_lander.ipynb) trains the Gymnasium Lunar Lander environment, then uses the trained environment to run inference and produce an episode that can be ingested.

After all cells have run, select the file icon on the right side to open the `Files` panel. Hover over the `stable_baselines_example.json` file and use the three dots to open the menu and download the file.

You can then ingest the results file:

```bash
docker compose run --rm django ./manage.py ingest_episode /path/to/stable_baselines_example.json
```

Assuming that the sever is already running with `docker compose up` (see [setup docs](../../../README.md)), you can visit the [landing page](http://localhost:8000) where this episode should now be available.
