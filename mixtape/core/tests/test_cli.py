from pathlib import Path
from typing import Generator

from click.testing import CliRunner
from django.core.files import File
from django.db.models.signals import post_save
import pytest

from mixtape.core.management.commands.inference import inference as inference_command
from mixtape.core.models import Checkpoint, Episode, Inference, Training
from mixtape.core.models.episode import auto_compute_clustering


@pytest.fixture
def _inference_task(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Avoid Ray/RLlib by stubbing the Celery task body."""
    disconnected = post_save.disconnect(auto_compute_clustering, sender=Episode)

    def fake_run(*, inference_pk: int, **_: object) -> None:
        inference = Inference.objects.get(pk=inference_pk)
        Episode.objects.create(inference=inference)

    from mixtape.core.tasks.inference_tasks import run_inference_task

    monkeypatch.setattr(run_inference_task, 'run', fake_run)

    try:
        yield None
    finally:
        if disconnected:
            post_save.connect(auto_compute_clustering, sender=Episode, weak=False)


def _data_dir() -> Path:
    return Path(__file__).parent / 'data'


def _create_checkpoint_from_test_data(env_name: str, filename: str) -> Checkpoint:
    training = Training.objects.create(
        environment=env_name,
        algorithm='PPO',
        parallel=False,
        num_gpus=0.0,
        iterations=1,
        is_external=True,
    )

    archive_path = _data_dir() / filename
    with archive_path.open('rb') as archive_stream:
        return Checkpoint.objects.create(
            training=training,
            last=True,
            archive=File(archive_stream, name=f'checkpoint/{filename}'),
        )


@pytest.mark.parametrize(
    ('env_name', 'checkpoint_filename', 'cli_parallel'),
    [
        ('knights_archers_zombies_v10', 'kaz_checkpoint.tar.bz2', True),
        ('pistonball_v6', 'pistonball_checkpoint.tar.bz2', False),
        ('LunarLander-v2', 'lunar_lander_checkpoint.tar.bz2', False),
    ],
    ids=[
        'pettingzoo_parallel',
        'pettingzoo',
        'gymnasium',
    ],
)
@pytest.mark.django_db
def test_cli_inference(
    cli_runner: CliRunner,
    _inference_task: None,
    env_name: str,
    checkpoint_filename: str,
    cli_parallel: bool,
):
    checkpoint = _create_checkpoint_from_test_data(env_name, checkpoint_filename)

    inference_result = cli_runner.invoke(
        inference_command,
        [
            str(checkpoint.id),
            *(['--parallel'] if cli_parallel else []),
            '--immediate',
        ],
    )
    assert inference_result.exit_code == 0
    inference = Inference.objects.get(checkpoint=checkpoint)
    assert inference.parallel is cli_parallel
    assert Episode.objects.filter(inference__checkpoint=checkpoint).exists()
