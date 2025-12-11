from django.core.exceptions import ValidationError
from django.db import models

from mixtape.core.json_encoder import CustomJSONEncoder
from mixtape.core.ray_utils.constants import ExampleEnvs, SupportedAlgorithm


class Training(models.Model):
    created = models.DateTimeField(auto_now_add=True)

    environment = models.CharField(max_length=200)
    algorithm = models.CharField(max_length=200)
    parallel = models.BooleanField(default=False)
    num_gpus = models.FloatField(default=0.0)
    iterations = models.PositiveIntegerField(default=100)

    config = models.JSONField(default=dict, blank=True, null=True, encoder=CustomJSONEncoder)

    # Reward mapping for environments with multiple reward components
    reward_mapping = models.JSONField(null=True, blank=True)

    is_external = models.BooleanField(default=False)

    def clean(self):
        if not self.is_external:
            if self.environment not in ExampleEnvs.values:
                raise ValidationError({'environment': 'Invalid environment'})
            if self.algorithm not in SupportedAlgorithm.values:
                raise ValidationError({'algorithm': 'Invalid algorithm'})
