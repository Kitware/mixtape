from django.contrib import admin

from mixtape.core.models.inference_request import InferenceRequest


@admin.register(InferenceRequest)
class InferenceRequestRequestAdmin(admin.ModelAdmin):
    list_display = ['id', 'created', 'checkpoint', 'parallel', 'config']
