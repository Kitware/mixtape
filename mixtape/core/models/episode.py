from django.db import models

from .inference_request import InferenceRequest


class Episode(models.Model):
    inference_request = models.ForeignKey(InferenceRequest, on_delete=models.CASCADE)
