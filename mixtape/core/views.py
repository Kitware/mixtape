from collections import defaultdict
from itertools import accumulate

from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render

from mixtape.core.models.episode import Episode
from mixtape.core.ray_utils.utility_functions import get_environment_mapping



def _episode_insights(episode_pk: int, group_by_episode: bool = False) -> dict:
    # Prefetch all related data in a single query
    episode = get_object_or_404(
        Episode.objects.select_related('inference__checkpoint__training').prefetch_related(
            'steps', 'steps__agent_steps', 'steps__agent_steps__unit_steps'
        ),
        pk=episode_pk,
    )

    # Initialize data structures
    friendly_health_data: dict[str, list[float]] = {}
    enemy_health_data: dict[str, list[float]] = {}
    unit_actions_over_time: dict[str, list[dict]] = {}
    unit_lifetimes: dict[str, dict] = {}
    environment_rewards: list[float] = []
    value_estimates: dict[str, list[float]] = {}
    predicted_rewards: dict[str, list[float]] = {}
    total_rewards: dict[str, list[float]] = {}

    # Prepare step data and collect health data
    step_data: dict[int, dict] = {}
    for step in episode.steps.all().order_by('number'):
        step_environment_reward = 0.0
        step_data[step.number] = {
            'image_url': step.image.url if step.image else None,
            'agent_steps': [],
        }

        # Track unit actions for this step
        step_unit_actions = {}

        for agent_step in step.agent_steps.all():
            agent_key = agent_step.agent

            # Track value estimates
            if agent_step.value_estimate is not None:
                if agent_key not in value_estimates:
                    value_estimates[agent_key] = []
                value_estimates[agent_key].append(agent_step.value_estimate)

            # Track value estimates
            if agent_step.value_estimate is not None:
                if agent_key not in value_estimates:
                    value_estimates[agent_key] = []
                value_estimates[agent_key].append(agent_step.value_estimate)

            # Track predicted rewards if available
            if agent_step.custom_metrics and 'predicted_rewards' in agent_step.custom_metrics:
                if agent_key not in predicted_rewards:
                    predicted_rewards[agent_key] = []
                predicted_rewards[agent_key].append(agent_step.custom_metrics['predicted_rewards'])

            # Initialize total rewards tracking
            if agent_key not in total_rewards:
                total_rewards[agent_key] = []
            total_rewards[agent_key].append(agent_step.total_reward)

            # Sum rewards for environment reward
            step_environment_reward += agent_step.total_reward

            # Process agent health
            if agent_step.health is not None:
                if agent_key not in friendly_health_data:
                    friendly_health_data[agent_key] = []
                friendly_health_data[agent_key].append(agent_step.health)

            # Process enemy agent health
            if agent_step.enemy_agent_health:
                for i, enemy_health in enumerate(agent_step.enemy_agent_health):
                    enemy_agent_key = f'enemy_{agent_key}'
                    print(f"enemy_agent_key: {enemy_agent_key}, {i}, {enemy_health}, {agent_key}")
                    if enemy_agent_key not in enemy_health_data:
                        enemy_health_data[enemy_agent_key] = []
                    enemy_health_data[enemy_agent_key].append(enemy_health)

            # Process enemy unit health (rename to enemy_1..enemy_5)
            if agent_step.enemy_unit_health:
                for i, enemy_health in enumerate(agent_step.enemy_unit_health):
                    enemy_unit_key = f'enemy_{i+1}'
                    if enemy_unit_key not in enemy_health_data:
                        enemy_health_data[enemy_unit_key] = []
                    enemy_health_data[enemy_unit_key].append(enemy_health)

            # Process unit steps
            for unit_step in agent_step.unit_steps.all():
                unit_key = unit_step.unit

                # Track unit actions
                if unit_key not in step_unit_actions:
                    action_value = None if unit_step.action == -1 else unit_step.action
                    step_unit_actions[unit_key] = {
                        'action': action_value,
                        'is_alive': unit_step.action != -1,
                        'reward': unit_step.total_reward,
                        'health': unit_step.health,
                        'unit': unit_key,
                        'action_name': unit_step.action_string,
                    }

                # Track health data
                if unit_step.health is not None:
                    if unit_key not in friendly_health_data:
                        friendly_health_data[unit_key] = []
                    friendly_health_data[unit_key].append(unit_step.health)

                # Track unit predicted rewards if available
                if unit_step.custom_metrics and 'predicted_rewards' in unit_step.custom_metrics:
                    unit_pred_key = f'unit_{unit_key}'
                    if unit_pred_key not in predicted_rewards:
                        predicted_rewards[unit_pred_key] = []
                    predicted_rewards[unit_pred_key].append(
                        unit_step.custom_metrics['predicted_rewards']
                    )

                # Track unit total rewards
                unit_total_key = f'unit_{unit_key}'
                if unit_total_key not in total_rewards:
                    total_rewards[unit_total_key] = []
                total_rewards[unit_total_key].append(sum(unit_step.rewards))

            # Add agent and unit steps to step_data
            step_data[step.number]['agent_steps'].append(
                {
                    'agent': agent_key,
                    'action': agent_step.action_string,
                    'total_reward': agent_step.total_reward,
                    'value_estimate': agent_step.value_estimate,
                    'health': agent_step.health,
                    'unit_steps': step_unit_actions,
                }
            )

        # Store environment reward for this step
        environment_rewards.append(step_environment_reward)

        # Update unit actions over time
        for unit_id, action_data in step_unit_actions.items():
            if unit_id not in unit_actions_over_time:
                unit_actions_over_time[unit_id] = []
            unit_actions_over_time[unit_id].append(
                {
                    'step': step.number,
                    'action': action_data['action'],
                    'is_alive': action_data['is_alive'],
                }
            )

    # Calculate unit lifetimes based on first death action (None, which was -1)
    for unit_id, actions in unit_actions_over_time.items():
        death_step = None
        for action in actions:
            if action['action'] is None:
                death_step = action['step']
                break
        unit_lifetimes[unit_id] = {
            'death_step': death_step,
            'total_steps': (
                death_step if death_step else (actions[-1]['step'] + 1 if actions else 0)
            ),
        }

    # Prepare plot data
    env_name = episode.inference.checkpoint.training.environment
    training = episode.inference.checkpoint.training

    # Get unit mapping for the environment
    unit_mapping = get_environment_mapping(env_name).get('unit_mapping', {})

    reward_histogram = [
        a.total_reward for step in episode.steps.all() for a in step.agent_steps.all()
    ]

    action_map = get_environment_mapping(env_name)
    action_v_reward: defaultdict[str, float] | defaultdict[str, defaultdict[str, float]] = (
        defaultdict(float if group_by_episode else lambda: defaultdict(float))  # type: ignore
    )
    action_v_frequency: defaultdict[str, int] | defaultdict[str, defaultdict[str, int]] = (
        defaultdict(int if group_by_episode else lambda: defaultdict(int))  # type: ignore
    )

    unique_agents = set()
    reward_mapping = training.reward_mapping or []
    unit_navigation_rewards: dict[str, list[float]] = {}
    unit_combat_rewards: dict[str, list[float]] = {}

    for step in episode.steps.all().prefetch_related('agent_steps__unit_steps'):
        step_unit_rewards: dict[str, dict[str, float]] = {}

        # Process agent steps
        for agent_step in step.agent_steps.all():
            unique_agents.add(agent_step.agent)
            action = action_map.get(f'{int(agent_step.action)}', f'{agent_step.action}')
            key = action if group_by_episode else agent_step.agent
            if group_by_episode:
                action_v_reward[action] += agent_step.total_reward  # type: ignore
                action_v_frequency[action] += 1  # type: ignore
            else:
                action_v_reward[key][action] += agent_step.total_reward  # type: ignore
                action_v_frequency[key][action] += 1  # type: ignore


            # Process unit steps for this agent step
            for unit_step in agent_step.unit_steps.all():
                for i, reward_name in enumerate(reward_mapping):
                    if i < len(unit_step.rewards):
                        unit_info = step_unit_rewards.setdefault(unit_step.unit, {})
                        unit_info.setdefault(reward_name, 0)
                        step_unit_rewards[unit_step.unit][reward_name] += unit_step.rewards[i]

        # Update unit rewards for this step
        for unit, rewards in step_unit_rewards.items():
            # Handle both 'navigation'/'nav_reward' naming
            nav_key = None
            for key in ['navigation', 'nav_reward']:
                if key in rewards:
                    nav_key = key
                    break

            if nav_key:
                unit_navigation_rewards.setdefault(unit, [])
                unit_navigation_rewards[unit].append(rewards[nav_key])

            # Handle both 'combat'/'combat_reward' naming
            combat_key = None
            for key in ['combat', 'combat_reward']:
                if key in rewards:
                    combat_key = key
                    break

            if combat_key:
                unit_combat_rewards.setdefault(unit, [])
                unit_combat_rewards[unit].append(rewards[combat_key])


    # Calculate cumulative navigation rewards for units
    unit_cumulative_navigation: dict[str, list[float]] = {}
    for unit in unit_navigation_rewards.keys():
        nav_rewards: list[float] = unit_navigation_rewards[unit]
        unit_cumulative_navigation[unit] = list(accumulate(nav_rewards))

    # Calculate cumulative combat rewards for units
    unit_cumulative_combat: dict[str, list[float]] = {}
    for unit in unit_combat_rewards.keys():
        combat_rewards: list[float] = unit_combat_rewards[unit]
        unit_cumulative_combat[unit] = list(accumulate(combat_rewards))

    # Used to populate the timeline
    key_steps = episode.steps.all().order_by('number')

    # In Python since total_reward is a property
    key_steps_values = []
    for step in key_steps:
        key_steps_values.append(
            {
                'number': step.number,
                'total_rewards': sum(
                    agent_step.total_reward for agent_step in step.agent_steps.all()
                ),
            }
        )

    # TODO: Revist this. This exists more as a placeholder, timeline
    #       should represent points of interest with more meaning.
    # Get top 40 steps by total rewards, ordered by step number
    timeline_steps = sorted(
        [step for step in key_steps_values if step['total_rewards'] > 0],
        key=lambda x: x['total_rewards'],
        reverse=True,
    )[:15]
    timeline_steps = sorted(timeline_steps, key=lambda x: x['number'])
    timeline_steps_serialized = [
        {
            'number': step['number'],
            'total_rewards': step['total_rewards'],
        }
        for step in timeline_steps
    ]

    return {
        'episode_details': episode,
        'action_v_reward': action_v_reward,
        'reward_histogram': reward_histogram,
        'action_v_frequency': action_v_frequency,
        'timeline_key_steps': timeline_steps_serialized,
        'reward_mapping': reward_mapping,
        'step_data': step_data,
        'unique_agents': list(unique_agents),
        'friendly_health_data': friendly_health_data,
        'enemy_health_data': enemy_health_data,
        'unit_actions': unit_actions_over_time,
        'unit_lifetimes': unit_lifetimes,
        'environment_rewards': environment_rewards,
        'value_estimates': value_estimates,
        'unit_mapping': unit_mapping,
        'unit_navigation_rewards': unit_cumulative_navigation,
        'unit_combat_rewards': unit_cumulative_combat,
        'predicted_rewards': predicted_rewards,
        'total_rewards': total_rewards,
    }


def insights(request: HttpRequest) -> HttpResponse:
    # Get episode IDs from query parameters
    episode_ids = request.GET.getlist('episode_id')
    if not episode_ids:
        raise Http404('No episode IDs provided')
    try:
        episode_pks = [int(episode_id) for episode_id in episode_ids]
    except ValueError:
        raise Http404('Invalid episode ID format')

    # Process each episode and combine the data
    all_episode_details: list[Episode] = []
    all_action_v_reward: dict[str, dict[str, float]] = {}
    all_reward_histogram: list[float] = []
    all_action_v_frequency: dict[str, dict[str, int]] = {}
    all_step_data: list[dict] = []
    all_timeline_steps: list[dict] = []
    all_friendly_health_data: list[dict] = []
    all_enemy_health_data: list[dict] = []
    all_unit_actions: list[dict] = []
    all_unit_lifetimes: list[dict] = []
    all_environment_rewards: list[list[float]] = []
    all_value_estimates: list[dict] = []
    all_unit_navigation_rewards: list[dict] = []
    all_unit_combat_rewards: list[dict] = []
    all_predicted_rewards: list[dict] = []
    all_total_rewards: list[dict] = []

    group_by_episode = len(episode_pks) > 1
    for episode_pk in episode_pks:
        insight_results = _episode_insights(episode_pk, group_by_episode)
        all_episode_details.append(insight_results['episode_details'])
        all_action_v_reward[f'Episode {episode_pk}'] = insight_results['action_v_reward']
        all_reward_histogram.append(insight_results['reward_histogram'])
        all_action_v_frequency[f'Episode {episode_pk}'] = insight_results['action_v_frequency']
        all_step_data.append(insight_results['step_data'])
        all_timeline_steps.append(insight_results['timeline_key_steps'])
        all_friendly_health_data.append(insight_results['friendly_health_data'])
        all_enemy_health_data.append(insight_results['enemy_health_data'])
        all_unit_actions.append(insight_results['unit_actions'])
        all_unit_lifetimes.append(insight_results['unit_lifetimes'])
        all_environment_rewards.append(insight_results['environment_rewards'])
        all_value_estimates.append(insight_results['value_estimates'])
        all_unit_navigation_rewards.append(insight_results['unit_navigation_rewards'])
        all_unit_combat_rewards.append(insight_results['unit_combat_rewards'])
        all_predicted_rewards.append(insight_results['predicted_rewards'])
        all_total_rewards.append(insight_results['total_rewards'])

    # Get unique agents across all episodes
    unique_agents = insight_results['unique_agents']


    data = {
        'all_episode_details': all_episode_details,
        'parsed_data': {
            'action_v_reward': all_action_v_reward,
            'reward_histogram': all_reward_histogram,
            'action_v_frequency': all_action_v_frequency,
            'unique_agents': unique_agents,
            'episode_ids': episode_pks,
            'max_steps': max(len(step_data) for step_data in all_step_data),
            'steps': all_step_data,
            'friendly_health_data': all_friendly_health_data,
            'enemy_health_data': all_enemy_health_data,
            'unit_actions': all_unit_actions,
            'unit_lifetimes': all_unit_lifetimes,
            'environment_rewards': all_environment_rewards,
            'value_estimates': all_value_estimates,
            'step_data': all_step_data,
            'timeline_key_steps': all_timeline_steps,
            'unit_navigation_rewards': all_unit_navigation_rewards,
            'unit_combat_rewards': all_unit_combat_rewards,
            'predicted_rewards': all_predicted_rewards,
            'total_rewards': all_total_rewards,
        },
        'has_reward_mapping': any(
            episode.inference.checkpoint.training.reward_mapping for episode in all_episode_details
        ),
        'group_by_episode': group_by_episode,
    }

    return render(
        request,
        'core/insights.html',
        data,
    )


def home_page(request: HttpRequest) -> HttpResponse:
    episodes = Episode.objects.select_related('inference__checkpoint__training').all()
    algorithms = (
        episodes.values_list('inference__checkpoint__training__algorithm', flat=True)
        .distinct()
        .order_by('inference__checkpoint__training__algorithm')
    )
    environments = (
        episodes.values_list('inference__checkpoint__training__environment', flat=True)
        .distinct()
        .order_by('inference__checkpoint__training__environment')
    )
    return render(
        request,
        'core/home.html',
        {'episodes': episodes, 'algorithms': algorithms, 'environments': environments},
    )
