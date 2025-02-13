from django.contrib import admin

from mixtape.core.models.episode import Episode


@admin.register(Episode)
class EpisodeAdmin(admin.ModelAdmin):
    list_display = ['id', 'inference_request']
