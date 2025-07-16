from django.contrib import admin

from mixtape.core.models.agent_step import AgentStep


@admin.register(AgentStep)
class AgentStepAdmin(admin.ModelAdmin):
    list_display = ['id', 'agent', 'step', 'action', 'total_reward', 'observation_space']
    list_filter = ['agent', 'action']
