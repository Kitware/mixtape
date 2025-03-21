from collections import defaultdict
from itertools import accumulate
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
    agent_steps = AgentStep.objects.filter(step__episode_id=episode_pk)
    agent_steps_aggregation = agent_steps.annotate(
        total_rewards=Sum('reward'),
        reward_frequency=Count('reward'),
        action_frequency=Count('action'),
    ).order_by('action', 'reward')

    environment = episode.inference_request.checkpoint.training_request.environment
    plot_data: dict[str, Any] = {
        # dict mapping agent (str) to action (str) to total reward (float)
        'action_v_reward': defaultdict(lambda: defaultdict(float)),
        # all reward values received over the episode (list of floats)
        'reward_histogram': [a.reward for a in agent_steps],
        # dict mapping action (str) to freuency of action (int)
        'action_v_frequency': defaultdict(int),
    }

    action_map = actions.get(environment, {})
    for entry in agent_steps_aggregation:
        action = action_map.get(int(entry.action), f'{entry.action}')
        plot_data['action_v_reward'][entry.agent][action] += entry.total_rewards
        plot_data['action_v_frequency'][action] += entry.action_frequency

    key_steps = steps.annotate(total_rewards=Sum('agent_steps__reward', default=0)).order_by(
        'number'
    )
    plot_data['rewards_over_time'] = list(accumulate(ks.total_rewards for ks in key_steps))
    timeline_steps = key_steps.filter(total_rewards__gt=0)

    return render(
        request,
        'core/insights.html',
        {
            'episode': episode,
            'steps': steps,
            'plot_data': plot_data,
            'timeline_steps': timeline_steps,
        },
    )


def home_page(request: HttpRequest) -> HttpResponse:
    episodes = Episode.objects.all()
    return render(request, 'core/home.html', {'episodes': episodes})
