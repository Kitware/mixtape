from django.db import models
from django.db.models import Q

from .training_request import TrainingRequest


class Checkpoint(models.Model):
    class Meta:
        constraints = [
            # TODO: What if best / last is False? Should it be excluded from the constraint?
            models.UniqueConstraint(
                fields=['training_request', 'best'], name='unique_checkpoint_best'
            ),
            models.UniqueConstraint(
                fields=['training_request', 'last'], name='unique_checkpoint_last'
            ),
            models.CheckConstraint(condition=Q(best=True) | Q(last=True), name='best_or_last'),
        ]

    created = models.DateTimeField(auto_now_add=True)

    training_request = models.ForeignKey(
        TrainingRequest, on_delete=models.CASCADE, related_name='checkpoints'
    )
    best = models.BooleanField(default=False)
    last = models.BooleanField(default=False)
    archive = models.FileField()
