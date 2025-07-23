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
        Continuous action spaces should skip this.
    */
    "action_mapping": {
        "0": "string",
        "1": "string"
    },
    "training": {
        "environment": "string",
        "algorithm": "string",
        "parallel": false,                          // Optional, defaults to false
        "num_gpus": 0.0,                            // Optional
        "iterations": 100,
        "config": {},                               // Optional
        "reward_mapping": ["string"]                // Optional
    },
    "inference": {
        "parallel": false,                          // Optional, defaults to false
        "config": {},                               // Optional
        "steps": [                                  // Must have at least one step
            {
                "number": 0,                        // This is the step number
                "image": "base64_encoded_string",   // Optional
                "agent_steps": [                    // Optional, one agent_step per agent that performs an action in the current step
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
                        "action_distribution": ["float"]   // Optional
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
