from django.contrib import admin
from django.utils.html import format_html

from mixtape.core.models.step import Step


@admin.register(Step)
class StepAdmin(admin.ModelAdmin):
    list_display = ['id', 'episode', 'environment_name', 'image_thumbnail', 'number']
    list_filter = ['episode', 'episode__inference__checkpoint__training__environment']

    @admin.display(description='Environment')
    def environment_name(self, obj):
        return obj.episode.inference.checkpoint.training.environment

    @admin.display(description='Image')
    def image_thumbnail(self, obj):
        if obj.image and obj.image.url:
            return format_html(
                '<a href="{}" target="_blank"><img src="{}" width="100" height="auto" /></a>',
                obj.image.url,
                obj.image.url,
            )
        return '-'
