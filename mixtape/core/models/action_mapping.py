from django.db import models


class ActionMapping(models.Model):
    class Meta:
        constraints = [models.UniqueConstraint(fields=['environment'], name='unique_environment')]

    created = models.DateTimeField(auto_now_add=True)

    environment = models.CharField(max_length=200)
    mapping = models.JSONField()

    @classmethod
    def get_environment_mapping(cls, environment: str) -> dict:
        try:
            custom_mapping = cls.objects.get(environment=environment)
        except cls.DoesNotExist:
            return {}
        return custom_mapping.mapping
