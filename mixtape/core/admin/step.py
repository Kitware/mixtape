from django.contrib import admin

from mixtape.core.models.step import Step


@admin.register(Step)
class StepAdmin(admin.ModelAdmin):
    list_display = ['id', 'episode', 'image', 'number']
    list_filter = ['episode']
