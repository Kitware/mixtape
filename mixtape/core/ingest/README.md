# External Episode Ingestion

This directory contains functionality for ingesting external episodes into the Mixtape system. This document explains how to create a valid JSON file and how to ingest it.

## JSON File Structure

The JSON file is required to contain information about both the training as well as inference. The following is a breakdown of the required format:

```jsonc
{
    "agent_mapping": {                             // Optional, defaults to using integer action values
        "0": "string",
        "1": "string"
    },
    "training": {
        "environment": "string",                   // Required, max length 200 chars
        "algorithm": "string",                     // Required, max length 200 chars
        "parallel": false,                         // Optional, defaults to false
        "num_gpus": 0.0,                           // Optional, must be >= 0.0, defaults to 0.0
        "iterations": 100,                         // Required, must be >= 1
        "config": {}                               // Optional, defaults to empty dict
    },
    "inference": {
        "parallel": false,                         // Optional, defaults to false
        "config": {},                              // Optional, defaults to empty dict
        "steps": [                                 // Required, must have at least one step
            {
                "number": 0,                       // Required, must be >= 0
                "image": "base64_encoded_string",  // Optional, can be an empty string if no image is available
                "agent_steps": [                   // Optional, one agent_step per agent that performs an action in the current step
                    {
                        "agent": "string",         // Required, max length 200 chars
                        "action": 0.0,             // Required
                        "reward": 0.0,             // Required
                        "observation_space": [0.0] // Required, can be 1D or 2D array of floats
                    }
                ]
            }
        ]
    }
}
```

### Notes on Fields

- **Agent Mapping**:
  - Each action in the action space can be mapped to a string value. This is used in the UI for improved labeling of results.

- **Training**:
  - `environment` and `algorithm` are required string fields
  - `num_gpus` must be a non-negative number
  - `iterations` must be a positive integer
  - `config` can contain any additional training configuration information formatted as a Python dict

- **Inference**:
  - `steps` must contain at least one step
  - Each step must have a non-negative `number`
  - All steps should be present, even if there is no agent_step associated with it
  - `image` should be a base64-encoded string if provided
  - `agent_steps` is optional but when provided, each agent step must include all required fields

## Ingesting the JSON File

To ingest an external episode, use the Django management command:

```bash
python manage.py ingest_episode path/to/your/episode.json
```

The command will validate the JSON structure against the required schema and create all the required records.

## Errors

The ingest process will fail if:
- The JSON file is not formatted correctly
- Required values are missing
- Values don't meet validation requirements
- The base64-encoded image data is invalid

If any part of the ingestion fails, no partial data will be saved to the database.

## Example

The simple [Stable Baselines example](./simple_baselines_lunar_lander.ipynb) trains the Gymnasium Lunar Lander environment, then uses the trained environment to run inference and produce an episode that can be ingested.

After all cells have run, select the file icon on the right side to open the `Files` panel. Hover over the `stable_baselines_example.json` file and use the three dots to open the menu and download the file.

You can then ingest the results file:

```bash
python manage.py ingest_episode /path/to/stable_baselines_example.json
```

Visit the [landing page](http://localhost:8000) where this episode should now be available.
