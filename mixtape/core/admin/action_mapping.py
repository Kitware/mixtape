from django.contrib import admin

from mixtape.core.models.action_mapping import ActionMapping


@admin.register(ActionMapping)
class ActionMappingAdmin(admin.ModelAdmin):
    list_display = ['id', 'environment', 'mapping']
    list_filter = ['environment']
