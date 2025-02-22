from django.shortcuts import get_list_or_404, render

from mixtape.core.models.step import Step


def simple_view(request, episode_id):
    steps = get_list_or_404(
        Step.objects.filter(episode_id=episode_id).prefetch_related('agent_steps')
    )
    return render(request, 'core/simple_view.html', {'steps': steps})
