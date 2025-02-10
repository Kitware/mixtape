from django.contrib import admin

from mixtape.core.models import Checkpoint, TrainingRequest


@admin.register(TrainingRequest)
class TrainingRequestAdmin(admin.ModelAdmin):
    list_display = ['id', 'created', 'environment', 'algorithm', 'iterations', 'last_checkpoint']
    list_filter = ['environment', 'algorithm', 'last_checkpoint']  # admin.BooleanFieldListFilter

    @admin.display(description='Last checkpoint', boolean=True)
    def last_checkpoint(self, training_request: TrainingRequest) -> bool:
        return training_request.checkpoints.filter(last=True).exists()


@admin.register(Checkpoint)
class CheckpointAdmin(admin.ModelAdmin):
    list_display = ['id', 'training_request', 'created', 'last', 'archive']
