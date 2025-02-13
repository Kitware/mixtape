from django.contrib import admin

from mixtape.core.models import Checkpoint


@admin.register(Checkpoint)
class CheckpointAdmin(admin.ModelAdmin):
    list_display = ['id', 'training_request', 'created', 'last', 'archive']
