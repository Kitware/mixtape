import hashlib
import json
from typing import Optional

from django.contrib.postgres.fields import ArrayField
from django.db import models

from mixtape.core.analysis.clustering import DEFAULT_CLUSTERING_PARAMS
from mixtape.core.analysis.constants import FEATURE_TYPES

from .episode import Episode


class ClusteringResult(models.Model):
    class Status(models.TextChoices):
        SUCCESS = 'success', 'Success'
        FAILED = 'failed', 'Failed'

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['episode', 'feature_types', 'parameters_hash'],
                name='unique_episode_clustering_params',
            )
        ]

    episode = models.ForeignKey(
        Episode, on_delete=models.CASCADE, related_name='clustering_results'
    )
    feature_types = ArrayField(
        models.CharField(max_length=20, choices=[(ft, ft.title()) for ft in FEATURE_TYPES])
    )
    parameters = models.JSONField()
    parameters_hash = models.CharField(max_length=64, db_index=True)
    results = models.JSONField()
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.SUCCESS)
    error_message = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        """Auto-generate parameters hash on save."""
        if not self.parameters_hash:
            params_str = json.dumps(self.parameters, sort_keys=True)
            self.parameters_hash = hashlib.sha256(params_str.encode()).hexdigest()
        super().save(*args, **kwargs)

    @classmethod
    def get_for_episode_and_type(
        cls, episode_id: int, feature_type: str, parameters: Optional[dict] = None
    ):
        """Get clustering result for episode and feature type with optional parameters."""
        if parameters is None:
            parameters = DEFAULT_CLUSTERING_PARAMS

        params_str = json.dumps(parameters, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()

        return cls.objects.filter(
            episode_id=episode_id,
            feature_types__contains=[feature_type],
            parameters_hash=params_hash,
            status=cls.Status.SUCCESS,
        ).first()
