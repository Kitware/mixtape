from django.contrib import admin

from mixtape.core.models.rendered_step import RenderedStep


@admin.register(RenderedStep)
class RenderedStepAdmin(admin.ModelAdmin):
    list_display = ['id', 'episode', 'image', 'step']
    list_filter = ['episode']
