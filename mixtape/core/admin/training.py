from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html

from mixtape.core.models import Training


class CompletedListFilter(admin.SimpleListFilter):
    title = 'completed'
    parameter_name = 'completed'

    def lookups(self, request, model_admin):
        return [
            ('True', 'Complete'),
            ('False', 'Incomplete'),
        ]

    def queryset(self, request, queryset):
        if self.value() == 'True':
            return queryset.filter(checkpoints__isnull=False)
        elif self.value() == 'False':
            return queryset.filter(checkpoints__isnull=True)
        return queryset


@admin.register(Training)
class TrainingAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'created',
        'environment',
        'algorithm',
        'iterations',
        'completed',
        'last_checkpoint',
        'parallel',
    ]
    list_filter = [
        'environment',
        'algorithm',
        'parallel',
        CompletedListFilter,
    ]

    @admin.display(description='Completed', boolean=True, ordering='checkpoints')
    def completed(self, training: Training) -> bool:
        return training.checkpoints.exists()

    @admin.display(description='Last Checkpoint', ordering='checkpoints')
    def last_checkpoint(self, training: Training):
        last_checkpoint = training.checkpoints.filter(last=True).first()
        if last_checkpoint:
            last_checkpoint_url = reverse(
                'admin:core_checkpoint_change', kwargs={'object_id': last_checkpoint.pk}
            )
            return format_html('<a href="{}">Edit {}</a>', last_checkpoint_url, last_checkpoint.pk)
        else:
            return '-'

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.prefetch_related('checkpoints')
