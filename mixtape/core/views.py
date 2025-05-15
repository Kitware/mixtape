from collections import defaultdict
from itertools import accumulate
from typing import Any

from django.db.models import Subquery, Sum
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render

from mixtape.core.models.episode import Episode
from mixtape.core.ray_utils.utility_functions import get_environment_mapping


def insights(request: HttpRequest, episode_pk: int) -> HttpResponse:
    # Prefetch all related data in a single query
    episode = get_object_or_404(
        Episode.objects.select_related('inference__checkpoint__training').prefetch_related(
            'steps', 'steps__agent_steps'
        ),
        pk=episode_pk,
    )

    # Prepare step data
    step_data = {
        step.number: {
            'image_url': step.image.url if step.image else None,
            'agent_steps': [
                {
                    'agent': agent_step.agent,
                    'action': agent_step.action_string,
                    'reward': agent_step.reward,
                }
                for agent_step in step.agent_steps.all()
            ],
        }
        for step in episode.steps.all()
    }

    # Prepare plot data
    env_name = episode.inference.checkpoint.training.environment
    plot_data: dict[str, Any] = {
        # dict mapping agent (str) to action (str) to total reward (float)
        'action_v_reward': defaultdict(lambda: defaultdict(float)),
        # all reward values received over the episode (list of floats)
        'reward_histogram': [
            a.reward for step in episode.steps.all() for a in step.agent_steps.all()
        ],
        # dict mapping agent (str) to action (str) to frequency of action (int)
        'action_v_frequency': defaultdict(lambda: defaultdict(int)),
    }
    action_map = get_environment_mapping(env_name)
    for step in episode.steps.all():
        for agent_step in step.agent_steps.all():
            action = action_map.get(f'{int(agent_step.action)}', f'{agent_step.action}')
            plot_data['action_v_reward'][agent_step.agent][action] += agent_step.reward
            plot_data['action_v_frequency'][agent_step.agent][action] += 1
    plot_data['unique_agents'] = list(plot_data['action_v_reward'].keys())

    key_steps = (
        episode.steps.all()
        .annotate(total_rewards=Sum('agent_steps__reward', default=0))
        .order_by('number')
    )
    # list of cumulative rewards over time
    plot_data['rewards_over_time'] = list(accumulate(ks.total_rewards for ks in key_steps))
    # TODO: Revist this. This exists more as a placeholder, timeline
    #       should represent points of interest with more meaning.
    # Get top 40 steps by total rewards, ordered by step number
    timeline_steps = key_steps.filter(
        id__in=Subquery(
            key_steps.filter(total_rewards__gt=0).order_by('-total_rewards').values('id')[:40]
        )
    ).order_by('number')

    return render(
        request,
        'core/insights.html',
        {
            'episode': episode,
            'steps': episode.steps.all(),
            'plot_data': plot_data,
            'timeline_steps': timeline_steps,
            'step_data': step_data,
        },
    )


def home_page(request: HttpRequest) -> HttpResponse:
    episodes = Episode.objects.select_related('inference__checkpoint__training').all()
    return render(request, 'core/home.html', {'episodes': episodes})
