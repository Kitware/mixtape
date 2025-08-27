from django.contrib import admin

from mixtape.core.admin.utils import format_observation_space, pretty_format_dict
from mixtape.core.models.action_mapping import ActionMapping
from mixtape.core.models.episode_step import AgentStep


class ActionStringFilter(admin.SimpleListFilter):
    title = 'action'
    parameter_name = 'action_string'

    def lookups(self, request, model_admin):
        # Get all unique environment names from the database
        environments = set()
        for obj in model_admin.model.objects.select_related(
            'step__episode__inference__checkpoint__training'
        ).all():
            environments.add(obj.step.episode.inference.checkpoint.training.environment)

        # Get all action mappings for these environments
        action_values = set()
        for env in environments:
            try:
                mapping = ActionMapping.objects.get(environment=env).mapping
                for action_val, action_str in mapping.items():
                    action_values.add((action_val, f'{action_str} ({action_val})'))
            except ActionMapping.DoesNotExist:
                raise Exception(f'No action mapping found for environment: {env}')

        # Sort by the numeric action value
        return sorted(action_values, key=lambda x: float(x[0]))

    def queryset(self, request, queryset):
        value = self.value()
        if value is None:
            return queryset
        return queryset.filter(action=float(value))


@admin.register(AgentStep)
class AgentStepAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'agent',
        'step',
        'environment_name',
        'formatted_action',
        'formatted_rewards',
        'total_reward',
        'observation_shape',
        'formatted_health',
        'formatted_value_estimate',
        'formatted_predicted_reward',
        'formatted_custom_metrics',
        'formatted_action_distribution',
    ]
    list_filter = [
        'agent',
        ActionStringFilter,
        'step__episode__inference__checkpoint__training__environment',
    ]

    @admin.display(description='Environment')
    def environment_name(self, obj):
        return obj.step.episode.inference.checkpoint.training.environment

    @admin.display(description='Observation Shape')
    def observation_shape(self, obj):
        return format_observation_space(obj.observation_space)

    @admin.display(description='Health')
    def formatted_health(self, obj):
        return pretty_format_dict(obj.health)

    @admin.display(description='Custom Metrics')
    def formatted_custom_metrics(self, obj):
        return pretty_format_dict(obj.custom_metrics)

    @admin.display(description='Action Distribution')
    def formatted_action_distribution(self, obj):
        return pretty_format_dict(obj.action_distribution)

    @admin.display(description='Rewards')
    def formatted_rewards(self, obj):
        return pretty_format_dict(obj.rewards)

    @admin.display(description='Value Estimate')
    def formatted_value_estimate(self, obj):
        if obj.value_estimate is None:
            return '-'
        return round(obj.value_estimate, 2)

    @admin.display(description='Predicted Reward')
    def formatted_predicted_reward(self, obj):
        if obj.predicted_reward is None:
            return '-'
        return round(obj.predicted_reward, 2)

    @admin.display(description='Action')
    def formatted_action(self, obj):
        try:
            environment = obj.step.episode.inference.checkpoint.training.environment
            return obj._action_string(environment)
        except Exception:
            return f'{obj.action}'
