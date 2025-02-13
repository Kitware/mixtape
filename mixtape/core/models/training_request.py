from django.db import models

from mixtape.core.ray_utils.constants import ExampleEnvs, SupportedAlgorithm


class TrainingRequest(models.Model):
    created = models.DateTimeField(auto_now_add=True)

    environment = models.CharField(max_length=200, choices=ExampleEnvs)
    algorithm = models.CharField(max_length=200, choices=SupportedAlgorithm)
    parallel = models.BooleanField()
    num_gpus = models.FloatField(default=0.0)
    iterations = models.PositiveIntegerField(default=100)

    config = models.JSONField()
