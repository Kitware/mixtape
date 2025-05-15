from django.db import models

from mixtape.core.ray_utils.json_encoder import CustomJSONEncoder


class ActionMapping(models.Model):
    class Meta:
        constraints = [models.UniqueConstraint(fields=['environment'], name='unique_environment')]

    created = models.DateTimeField(auto_now_add=True)

    environment = models.CharField(max_length=200)
    mapping = models.JSONField(encoder=CustomJSONEncoder)
