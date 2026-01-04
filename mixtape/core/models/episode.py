from typing import Any

from django.db import models, transaction
from django.db.models.signals import post_save
from django.dispatch import receiver

from .inference import Inference


class Episode(models.Model):
    inference = models.ForeignKey(Inference, on_delete=models.CASCADE)


@receiver(post_save, sender=Episode)
def auto_compute_clustering(
    sender: type[Episode], instance: Episode, created: bool, **_: Any
) -> None:
    """Compute clustering when new episodes are created."""
    if created:
        # Avoid circular import
        from mixtape.core.tasks.clustering_tasks import compute_single_episode_clustering

        def queue_clustering():
            # Queue clustering computation for all feature types
            compute_single_episode_clustering.delay(instance.id)

        # Wait for transaction to commit before queuing the task
        transaction.on_commit(queue_clustering)
