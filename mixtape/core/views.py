import json

from django.db.models import Count, Sum
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render

from mixtape.core.models.agent_step import AgentStep
from mixtape.core.models.episode import Episode
from mixtape.core.models.step import Step


def insights(request: HttpRequest, episode_pk: int) -> HttpResponse:
    episode = get_object_or_404(Episode, pk=episode_pk)
    steps = Step.objects.prefetch_related('agent_steps').filter(episode=episode_pk)
    values = AgentStep.objects.filter(step__episode_id=episode_pk).values(
        'action', 'reward', 'agent'
    )
    results = values.annotate(
        total_rewards=Sum('reward'),
        reward_frequency=Count('reward'),
        action_frequency=Count('action'),
    ).order_by('action', 'reward')

    plot_data = {
        'action_v_reward': {},
        'reward_histogram': [v['reward'] for v in values],
        'action_v_frequency': {},
    }
    for entry in results:
        plot_data['action_v_reward'].setdefault(entry['agent'], {})
        plot_data['action_v_reward'][entry['agent']].setdefault(entry['action'], 0)
        plot_data['action_v_reward'][entry['agent']][entry['action']] += entry['total_rewards']
        plot_data['action_v_frequency'].setdefault(entry['action'], 0)
        plot_data['action_v_frequency'][entry['action']] += entry['action_frequency']

    return render(
        request,
        'core/insights.html',
        {'episode': episode, 'steps': steps, 'plot_data_json': json.dumps(plot_data)},
    )


def home_page(request: HttpRequest) -> HttpResponse:
    episodes = Episode.objects.all()
    return render(request, 'core/home.html', {'episodes': episodes})
