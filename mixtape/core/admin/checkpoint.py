from django.contrib import admin

from mixtape.core.models import Checkpoint


@admin.register(Checkpoint)
class CheckpointAdmin(admin.ModelAdmin):
    list_display = ['id', 'training', 'environment_name', 'created', 'last', 'archive']
    list_filter = ['training__environment', 'last']

    @admin.display(description='Environment')
    def environment_name(self, obj):
        return obj.training.environment
