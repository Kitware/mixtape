from django.db import models

from mixtape.core.json_encoder import CustomJSONEncoder

from .checkpoint import Checkpoint


class Inference(models.Model):
    created = models.DateTimeField(auto_now_add=True)

    checkpoint = models.ForeignKey(Checkpoint, on_delete=models.CASCADE)
    parallel = models.BooleanField(default=False)
    config = models.JSONField(default=dict, blank=True, null=True, encoder=CustomJSONEncoder)
