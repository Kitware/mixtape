from django.shortcuts import render

from mixtape.core.models.step import Step


def simple_view(request, episode_id):
    steps = Step.objects.filter(episode_id=episode_id)
    return render(request, 'simple_view.html', {'steps': steps})
