from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html
from django.utils.safestring import mark_safe

from mixtape.core.models.clustering_result import ClusteringResult


class ComputingStatusFilter(admin.SimpleListFilter):
    title = 'computation status'
    parameter_name = 'status'

    def lookups(self, request, model_admin):
        return [
            ('success', 'Completed'),
            ('failed', 'Failed'),
        ]

    def queryset(self, request, queryset):
        if self.value() == 'success':
            return queryset.filter(status=ClusteringResult.Status.SUCCESS)
        elif self.value() == 'failed':
            return queryset.filter(status=ClusteringResult.Status.FAILED)
        return queryset


@admin.register(ClusteringResult)
class ClusteringResultAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'episode_link',
        'feature_types_display',
        'status_display',
        'parameters_summary',
        'created_at',
        'updated_at',
    ]

    list_filter = [
        'feature_types',
        ComputingStatusFilter,
        'created_at',
        'updated_at',
    ]

    search_fields = [
        'episode__id',
        'parameters_hash',
    ]

    readonly_fields = [
        'parameters_hash',
        'created_at',
        'updated_at',
    ]

    fieldsets = [
        ('Episode & Features', {'fields': ['episode', 'feature_types']}),
        (
            'Parameters',
            {
                'fields': ['parameters', 'parameters_hash'],
                'classes': ['collapse'],
            },
        ),
        (
            'Results',
            {
                'fields': ['results', 'status', 'error_message'],
            },
        ),
        (
            'Timestamps',
            {
                'fields': ['created_at', 'updated_at'],
                'classes': ['collapse'],
            },
        ),
    ]

    @admin.display(description='Episode', ordering='episode')
    def episode_link(self, obj):
        """Link to the episode."""
        episode_url = reverse('admin:core_episode_change', kwargs={'object_id': obj.episode.pk})
        return format_html('<a href="{}">Episode {}</a>', episode_url, obj.episode.pk)

    @admin.display(description='Feature Types')
    def feature_types_display(self, obj):
        """Display feature types as comma-separated list."""
        return ', '.join(obj.feature_types)

    @admin.display(description='Status')
    def status_display(self, obj):
        """Display computation status with color coding."""
        if obj.status == ClusteringResult.Status.FAILED:
            return mark_safe('<span style="color: red;">Failed</span>')
        else:
            return mark_safe('<span style="color: green;">Completed</span>')

    @admin.display(description='Parameters')
    def parameters_summary(self, obj):
        """Display key parameters summary."""
        params = obj.parameters
        if isinstance(params, dict):
            key_params = []
            if 'kmeans_n_clusters' in params:
                key_params.append(f"clusters: {params['kmeans_n_clusters']}")
            if 'umap_n_components' in params:
                key_params.append(f"umap: {params['umap_n_components']}")
            return ', '.join(key_params) if key_params else 'Default'
        return 'Unknown'

    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        qs = super().get_queryset(request)
        return qs.select_related('episode')
