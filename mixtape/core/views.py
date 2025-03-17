import json
from typing import Any

from django.db.models import Count, Sum
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render

from mixtape.core.models.agent_step import AgentStep
from mixtape.core.models.episode import Episode
from mixtape.core.models.step import Step
from mixtape.environments.mappings import actions


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

    environment = episode.inference_request.checkpoint.training_request.environment
    plot_data: dict[str, Any] = {
        'action_v_reward': {},
        'reward_histogram': [v['reward'] for v in values],
        'action_v_frequency': {},
    }
    for entry in results:
        action = actions[environment][entry['action']]
        plot_data['action_v_reward'].setdefault(entry['agent'], {})
        plot_data['action_v_reward'][entry['agent']].setdefault(action, 0)
        plot_data['action_v_reward'][entry['agent']][action] += entry['total_rewards']
        plot_data['action_v_frequency'].setdefault(action, 0)
        plot_data['action_v_frequency'][action] += entry['action_frequency']

    key_steps = (
        steps.values('number')
        .annotate(total_rewards=Sum('agent_steps__reward', default=0))
        .order_by('number')
    )
    plot_data['rewards_over_time'] = []
    for i, ks in enumerate(key_steps):
        prev = plot_data['rewards_over_time'][i - 1] if i > 0 else 0
        plot_data['rewards_over_time'].append(ks['total_rewards'] + prev)

    return render(
        request,
        'core/insights.html',
        {
            'episode': episode,
            'steps': steps,
            'plot_data_json': json.dumps(plot_data),
            'key_steps': key_steps.filter(total_rewards__gt=0),
        },
    )


def home_page(request: HttpRequest) -> HttpResponse:
    episodes = Episode.objects.all()
    return render(request, 'core/home.html', {'episodes': episodes})
