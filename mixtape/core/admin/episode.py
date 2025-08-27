from django.contrib import admin

from mixtape.core.models.episode import Episode


@admin.register(Episode)
class EpisodeAdmin(admin.ModelAdmin):
    list_display = ['id', 'inference', 'environment_name', 'created']
    list_filter = ['inference__checkpoint__training__environment']

    @admin.display(description='Environment')
    def environment_name(self, obj):
        return obj.inference.checkpoint.training.environment

    @admin.display(description='Created')
    def created(self, obj):
        return obj.inference.created
