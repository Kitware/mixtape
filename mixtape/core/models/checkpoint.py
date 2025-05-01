from collections.abc import Generator
from contextlib import contextmanager
import json
from pathlib import Path, PurePath
from shutil import copyfileobj, make_archive, unpack_archive
from tempfile import NamedTemporaryFile, TemporaryDirectory

from django.core.files import File
from django.db import models
from django.db.models import Q

from .training import Training


class Checkpoint(models.Model):
    class Meta:
        constraints = [
            # TODO: What if best / last is False? Should it be excluded from the constraint?
            models.UniqueConstraint(
                fields=['training', 'best'], name='unique_checkpoint_best'
            ),
            models.UniqueConstraint(
                fields=['training', 'last'], name='unique_checkpoint_last'
            ),
        ]

    created = models.DateTimeField(auto_now_add=True)

    training = models.ForeignKey(
        Training, on_delete=models.CASCADE, related_name='checkpoints'
    )
    best = models.BooleanField(default=False)
    last = models.BooleanField(default=False)
    archive = models.FileField(null=True, blank=True)

    @contextmanager
    def archive_path(self) -> Generator[Path]:
        """Yield the archive as a directory on disk."""
        if not self.archive:
            raise ValueError('Checkpoint has no archive.')

        with TemporaryDirectory() as tmp_archive_dir:
            archive_dir = Path(tmp_archive_dir)
            with NamedTemporaryFile() as archive_file_stream:
                with self.archive.open('rb') as archive_stream:
                    copyfileobj(archive_stream, archive_file_stream)
                    archive_file_stream.seek(0)

                unpack_archive(archive_file_stream.name, archive_dir, format='bztar')

            # "state_file" within "rllib_checkpoint.json" contains an absolute path,
            # so rewrite it relative to this directory.
            with (archive_dir / 'rllib_checkpoint.json').open('r+') as metadata_file_stream:
                metadata = json.load(metadata_file_stream)
                metadata['state_file'] = str(archive_dir / PurePath(metadata['state_file']).name)
                metadata_file_stream.seek(0)
                json.dump(metadata, metadata_file_stream)
                metadata_file_stream.truncate()

            yield archive_dir

    @contextmanager
    @staticmethod
    def directory_to_file(directory: str, file_base_name: str) -> Generator[File]:
        """Archive a directory within a Django File."""
        with TemporaryDirectory() as archive_dir:
            archive_file = Path(
                make_archive(
                    base_name=str(Path(archive_dir) / 'archive'),
                    format='bztar',
                    root_dir=directory,
                )
            )
            with archive_file.open('rb') as archive_stream:
                yield File(archive_stream, name=f'{file_base_name}.tar.bz2')
