from django.contrib import admin

from mixtape.core.models.inference import Inference


@admin.register(Inference)
class InferenceAdmin(admin.ModelAdmin):
    list_display = ['id', 'created', 'checkpoint', 'parallel', 'config']
