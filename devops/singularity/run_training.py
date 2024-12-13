import argparse
import httpx


def build_params(**kwargs):
    """
    Build the URL parameters for the training endpoint
    """
    return {
        "parallel": kwargs.get("parallel", True),
        "timesteps_total": kwargs.get("timesteps_total", 5000),
        "num_gpus": kwargs.get("num_gpus", 0),
        "env_to_register": kwargs.get("env_to_register", "knights_archers_zombies_v10"),
    }


def build_body(**kwargs):
    """
    Build the body of the POST request
    """
    return {
        "env_config": kwargs.get("env_config", {}),
        "env_args": kwargs.get("env_args", {}),
        "training_args": kwargs.get("training_args", {}),
        "framework_args": kwargs.get("framework_args", {}),
        "run_args": kwargs.get("run_args", {}),
    }


def train_environment(port):
    # Construct the URL
    url = f"http://localhost:{port}/train"
    # Build the rquest parameters and body
    params = build_params()
    body = build_body()
    # Send the POST request to the server
    response = httpx.post(url, params=params, json=body, timeout=None)

    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Parse the port argument passed from the command line, defaulting to 5000
    parser.add_argument("-p", "--port", type=int, default=5000)
    args = parser.parse_args()

    # Call the training endpoint
    result = train_environment(args.port)
    print("Response from training:", result)
