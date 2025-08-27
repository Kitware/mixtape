from django.contrib import admin

from mixtape.core.admin.utils import pretty_format_dict
from mixtape.core.models.action_mapping import ActionMapping


@admin.register(ActionMapping)
class ActionMappingAdmin(admin.ModelAdmin):
    list_display = ['id', 'environment', 'formatted_mapping']
    list_filter = ['environment']

    @admin.display(description='Mapping', ordering='mapping')
    def formatted_mapping(self, obj):
        return pretty_format_dict(obj.mapping)
