import click

from mixtape.core.models.training import ExampleEnvs
from mixtape.core.ray_utils.environments import is_gymnasium_env


def check_parallel(
    ctx: click.Context, param: click.Option, value: bool | ExampleEnvs
) -> bool | ExampleEnvs:
    env_name = value if param.name == 'env_name' else ctx.params.get('env_name')
    parallel = value if param.name == 'parallel' else ctx.params.get('parallel', False)
    if env_name and is_gymnasium_env(env_name) and parallel:  # type: ignore
        click.echo(
            click.style(
                'Warning: The parallel option is only available for PettingZoo environments. '
                + 'Ignoring --parallel.',
                fg='red',
                bold=True,
            )
        )
        return False
    return value
