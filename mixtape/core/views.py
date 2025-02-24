from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render

from mixtape.core.models.episode import Episode
from mixtape.core.models.step import Step


def insights(request: HttpRequest, episode_pk: int) -> HttpResponse:
    episode = get_object_or_404(Episode, pk=episode_pk)
    steps = Step.objects.prefetch_related('agent_steps').filter(episode=episode_pk)
    return render(request, 'core/insights.html', {'episode': episode, 'steps': steps})


def home_page(request: HttpRequest) -> HttpResponse:
    episodes = Episode.objects.all()
    return render(request, 'core/home.html', {'episodes': episodes})
