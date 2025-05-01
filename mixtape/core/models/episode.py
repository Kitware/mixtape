from django.db import models

from .inference import Inference


class Episode(models.Model):
    inference = models.ForeignKey(Inference, on_delete=models.CASCADE)
