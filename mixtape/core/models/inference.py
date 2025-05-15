from django.db import models

from mixtape.core.ray_utils.json_encoder import CustomJSONEncoder

from .checkpoint import Checkpoint


class Inference(models.Model):
    created = models.DateTimeField(auto_now_add=True)

    checkpoint = models.ForeignKey(Checkpoint, on_delete=models.CASCADE)
    parallel = models.BooleanField()
    config = models.JSONField(default=dict, blank=True, null=True, encoder=CustomJSONEncoder)
