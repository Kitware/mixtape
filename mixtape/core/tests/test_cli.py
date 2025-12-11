from click.testing import CliRunner
import pytest

from mixtape.core.management.commands.training import training as training_command
from mixtape.core.models import Training


@pytest.mark.parametrize(
    ('env_name', 'parallel'),
    [
        ('knights_archers_zombies_v10', True),
        # ('knights_archers_zombies_v10', False),
        # ('BattleZone-v5', False),
    ],
    ids=[
        'pettingzoo_parallel',
        # 'pettingzoo_aec',
        # 'gymnasium',
    ],
)
@pytest.mark.django_db
def test_cli_training(cli_runner: CliRunner, env_name: str, parallel: bool):
    training_result = cli_runner.invoke(
        training_command,
        [
            '--env_name',
            env_name,
            '--algorithm',
            'PPO',
            *(['--parallel'] if parallel else []),
            '--training_iteration',
            '2',
            '--immediate',
        ],
    )
    assert training_result.exit_code == 0
    training = Training.objects.get()
    assert training.environment == env_name
    assert training.algorithm == 'PPO'
    assert training.parallel is parallel
    assert training.iterations == 2
