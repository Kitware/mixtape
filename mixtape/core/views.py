from django.db.models import Max
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render

from mixtape.core.models.episode import Episode
from mixtape.core.models.step import Step


def insights(request: HttpRequest, episode_pk: int) -> HttpResponse:
    episode = get_object_or_404(
        Episode.objects.annotate(step_count=Max('steps__number')), pk=episode_pk
    )
    return render(request, 'core/insights.html', {'episode': episode})


def insights_step(request: HttpRequest, episode_pk: int) -> HttpResponse:
    step_number = request.GET.get('step_number')
    step = get_object_or_404(
        Step.objects.prefetch_related('agent_steps'), episode=episode_pk, number=step_number
    )
    return render(request, 'core/insights_step.html', {'step': step})


def home_page(request: HttpRequest) -> HttpResponse:
    episodes = Episode.objects.all()
    return render(request, 'core/home.html', {'episodes': episodes})
