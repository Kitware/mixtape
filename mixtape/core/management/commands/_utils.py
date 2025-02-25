import click

from mixtape.core.ray_utils.environments import is_gymnasium_env


def check_parallel(ctx: click.Context, param: str, parallel: bool) -> bool:
    env_name = ctx.params['env_name']
    if is_gymnasium_env(env_name) and parallel:
        click.echo(
            click.style(
                'Warning: The parallel option is only available for PettingZoo environments. '
                + 'Ignoring --parallel.',
                fg='red',
                bold=True,
            )
        )
        return False
    return parallel
