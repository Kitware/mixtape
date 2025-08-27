from django.contrib import admin

from mixtape.core.models.inference import Inference


@admin.register(Inference)
class InferenceAdmin(admin.ModelAdmin):
    list_display = ['id', 'created', 'checkpoint', 'environment_name', 'parallel', 'has_config']
    list_filter = ['checkpoint__training__environment', 'parallel']

    @admin.display(description='Environment')
    def environment_name(self, obj):
        return obj.checkpoint.training.environment

    @admin.display(description='Config', boolean=True)
    def has_config(self, obj):
        return obj.config is not None and bool(obj.config)
