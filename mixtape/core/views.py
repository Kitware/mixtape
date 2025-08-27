from collections import defaultdict
from itertools import accumulate
import os

from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render
import einops
import numpy as np
from sklearn import cluster, decomposition, pipeline, preprocessing
import umap.umap_ as umap

from mixtape.core.models.episode import Episode
from mixtape.core.models.step import Step
from mixtape.core.ray_utils.utility_functions import get_environment_mapping


def _fetch_all_episode_features(episode_ids: list[int]) -> list[dict]:
    # Fetch all episodes and their steps
    all_steps = Step.objects.prefetch_related('agent_steps').filter(episode_id__in=episode_ids)

    # Group steps by episode
    episode_steps: dict[int, list[Step]] = {}
    for step in all_steps:
        if step.episode_id not in episode_steps:
            episode_steps[step.episode_id] = []
        episode_steps[step.episode_id].append(step)

    # Find the maximum number of timesteps across all episodes
    max_timesteps = 0
    for episode_id in episode_ids:
        max_timesteps = max(max_timesteps, len(episode_steps[episode_id]))

    # Get all unique agents across all episodes
    all_agent_steps = all_steps.exclude(agent_steps__agent=None)
    unique_agents = all_agent_steps.distinct('agent_steps__agent').values_list(
        'agent_steps__agent', flat=True
    )
    agent_to_idx = {agent: idx for idx, agent in enumerate(unique_agents)}
    n_agents = len(unique_agents)

    # Get observation space shape from first step
    first_step = episode_steps[episode_ids[0]][0]
    first_agent_step = first_step.agent_steps.first()
    if first_agent_step is None:
        raise Http404('No agent steps found for episode')
    first_obs = np.array(first_agent_step.observation_space).flatten()
    obs_shape = first_obs.shape
    if first_agent_step.action_distribution:
        first_action_dist = np.array(first_agent_step.action_distribution).flatten()
        action_dist_shape = first_action_dist.shape
    else:
        # If no action distribution data, create a default shape
        action_dist_shape = (1,)

    results = []

    for episode_id in episode_ids:
        steps = episode_steps[episode_id]

        # Initialize padded arrays
        obs = np.zeros((max_timesteps, n_agents, *obs_shape))
        agent_outs = np.zeros((max_timesteps, n_agents, *action_dist_shape))

        # Fill in the actual data
        for step in steps:
            step_idx = step.number
            for agent_step in step.agent_steps.all():
                if agent_step.agent in agent_to_idx:
                    agent_idx = agent_to_idx[agent_step.agent]
                    if agent_step.action_distribution:
                        agent_outs[step_idx, agent_idx] = np.array(
                            agent_step.action_distribution
                        ).flatten()
                    obs[step_idx, agent_idx] = np.array(agent_step.observation_space).flatten()

        results.append({'obs': obs, 'agent_outs': agent_outs})

    return results


def _cluster_episodes_by_feature(
    episodes: list[dict],
    feature_name: str,
    manifold_pipeline: pipeline.Pipeline,
    cluster_pipeline: cluster.KMeans,
    dimension_keys: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Stack all episodes' features along the episode dimension
    episode_features = np.stack([episode[feature_name] for episode in episodes])

    feature_shape = einops.parse_shape(
        episode_features,
        f'{dimension_keys} _',
    )

    all_features = einops.rearrange(
        episode_features,
        f'{dimension_keys} dim -> ({dimension_keys}) dim',
    )
    all_manifolds = manifold_pipeline.fit_transform(all_features)

    all_clusters = cluster_pipeline.fit_predict(all_manifolds)

    episode_clusters = einops.rearrange(
        all_clusters,
        f'({dimension_keys}) -> {dimension_keys}',
        **feature_shape,
    )
    episode_manifolds = einops.rearrange(
        all_manifolds,
        f'({dimension_keys}) dim -> {dimension_keys} dim',
        **feature_shape,
    )

    return episode_clusters, episode_manifolds, all_clusters, all_manifolds


def cluster_episodes_all_features(
    episode_ids: list[int],
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.5,
    umap_n_components: int = 20,
    pca_n_components: int = 2,
    kmeans_n_clusters: int = 10,
    feature_dimensions: str = 'episode time agent',
    seed: int = 42,
) -> tuple[dict, dict]:
    # Fetch both types of data
    features = _fetch_all_episode_features(episode_ids)
    obs_episodes = [{'obs': feat['obs']} for feat in features if 'obs' in feat]
    agent_outs_episodes = [
        {'agent_outs': feat['agent_outs']} for feat in features if 'agent_outs' in feat
    ]

    # Create pipelines
    avail_cpus = (os.cpu_count() or 0) - 1
    manifold_pipeline = pipeline.Pipeline(
        [
            ('scale', preprocessing.StandardScaler()),
            (
                'manifold',
                umap.UMAP(
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    n_components=umap_n_components,
                    densmap=False,
                    n_jobs=min(8, avail_cpus),
                ),
            ),
            ('decompose', decomposition.PCA(pca_n_components)),
        ]
    )
    cluster_pipeline = cluster.KMeans(kmeans_n_clusters, random_state=seed)

    # Cluster observations
    obs_clusters, obs_manifolds, obs_all_clusters, obs_all_manifolds = _cluster_episodes_by_feature(
        obs_episodes,
        'obs',
        manifold_pipeline,
        cluster_pipeline,
        dimension_keys=feature_dimensions,
    )

    # Cluster action distributions
    agent_outs_clusters, agent_outs_manifolds, agent_outs_all_clusters, agent_outs_all_manifolds = (
        _cluster_episodes_by_feature(
            agent_outs_episodes,
            'agent_outs',
            manifold_pipeline,
            cluster_pipeline,
            dimension_keys=feature_dimensions,
        )
    )

    obs_results = {
        'episode_clusters': obs_clusters,
        'episode_manifolds': obs_manifolds,
        'all_clusters': obs_all_clusters,
        'all_manifolds': obs_all_manifolds,
    }

    agent_outs_results = {
        'episode_clusters': agent_outs_clusters,
        'episode_manifolds': agent_outs_manifolds,
        'all_clusters': agent_outs_all_clusters,
        'all_manifolds': agent_outs_all_manifolds,
    }

    return obs_results, agent_outs_results


def _episode_insights(episode_pk: int, group_by_episode: bool = False) -> dict:
    # Prefetch all related data in a single query
    episode = get_object_or_404(
        Episode.objects.select_related('inference__checkpoint__training').prefetch_related(
            'steps', 'steps__agent_steps', 'steps__agent_steps__unit_steps'
        ),
        pk=episode_pk,
    )

    # Initialize data structures
    friendly_health_data = {}
    enemy_health_data = {}
    unit_actions_over_time = {}
    unit_lifetimes = {}
    environment_rewards = []
    value_estimates = {}

    # Prepare step data and collect health data
    step_data = {}
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

            # Sum rewards for environment reward
            step_environment_reward += agent_step.total_reward

            # Process agent health
            if agent_step.health and 'friendly' in agent_step.health:
                if agent_key not in friendly_health_data:
                    friendly_health_data[agent_key] = []
                friendly_health_data[agent_key].append(agent_step.health['friendly'])
            if agent_step.health and 'enemy' in agent_step.health:
                if agent_key not in enemy_health_data:
                    enemy_health_data[agent_key] = []
                enemy_health_data[agent_key].append(agent_step.health['enemy'])

            # Process unit steps
            for unit_step in agent_step.unit_steps.all():
                unit_key = unit_step.unit

                # Track unit actions
                if unit_key not in step_unit_actions:
                    step_unit_actions[unit_key] = {
                        'action': unit_step.action,
                        'is_alive': unit_step.action != -1,
                    }

                # Track health data
                if unit_step.health and 'friendly' in unit_step.health:
                    if unit_key not in friendly_health_data:
                        friendly_health_data[unit_key] = []
                    friendly_health_data[unit_key].append(unit_step.health['friendly'])
                if unit_step.health and 'enemy' in unit_step.health:
                    if unit_key not in enemy_health_data:
                        enemy_health_data[unit_key] = []
                    enemy_health_data[unit_key].append(unit_step.health['enemy'])

            # Add agent step to step_data
            step_data[step.number]['agent_steps'].append(
                {
                    'agent': agent_key,
                    'action': agent_step.action_string,
                    'total_reward': agent_step.total_reward,
                    'value_estimate': agent_step.value_estimate,
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

    # Calculate unit lifetimes based on first death action (-1)
    for unit_id, actions in unit_actions_over_time.items():
        death_step = None
        for action in actions:
            if action['action'] == -1:
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
    decomposed_rewards: dict[str, list[float]] = {v: [] for v in reward_mapping}
    unit_navigation_rewards = defaultdict(lambda: [])
    unit_combat_rewards = defaultdict(lambda: [])

    for step in episode.steps.all().prefetch_related('agent_steps__unit_steps'):
        step_decomposed_rewards = {v: 0.0 for v in reward_mapping}
        step_unit_rewards = defaultdict(lambda: {v: 0.0 for v in reward_mapping})

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

            # Process agent rewards
            for i, reward_name in enumerate(reward_mapping):
                if i < len(agent_step.rewards):
                    step_decomposed_rewards[reward_name] += agent_step.rewards[i]

            # Process unit steps for this agent step
            for unit_step in agent_step.unit_steps.all():
                for i, reward_name in enumerate(reward_mapping):
                    if i < len(unit_step.rewards):
                        step_unit_rewards[unit_step.unit][reward_name] += unit_step.rewards[i]

        # Update unit rewards for this step
        for unit, rewards in step_unit_rewards.items():
            if 'navigation' in rewards:
                unit_navigation_rewards[unit].append(rewards['navigation'])
            if 'combat' in rewards:
                unit_combat_rewards[unit].append(rewards['combat'])
            elif 'navigation' not in rewards:  # Only append 0 if neither reward exists yet
                unit_navigation_rewards[unit].append(0.0)

        # Update main rewards
        for reward_name in reward_mapping:
            decomposed_rewards[reward_name].append(step_decomposed_rewards[reward_name])

    # Calculate cumulative decomposed rewards
    cumulative_decomposed_rewards = {}
    for reward_name, rewards_list in decomposed_rewards.items():
        cumulative_decomposed_rewards[reward_name] = list(accumulate(rewards_list))

    # Calculate cumulative navigation rewards for units
    unit_cumulative_navigation = {}
    for unit, rewards in unit_navigation_rewards.items():
        unit_cumulative_navigation[unit] = list(accumulate(rewards))

    # Calculate cumulative combat rewards for units
    unit_cumulative_combat = {}
    for unit, rewards in unit_combat_rewards.items():
        unit_cumulative_combat[unit] = list(accumulate(rewards))

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

    # list of cumulative rewards over time
    rewards_over_time = list(accumulate(ks['total_rewards'] for ks in key_steps_values))
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
        'rewards_over_time': rewards_over_time,
        'decomposed_rewards': cumulative_decomposed_rewards,
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
    all_action_v_reward: defaultdict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    all_reward_histogram: list[float] = []
    all_action_v_frequency: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    all_rewards_over_time: list[int] = []
    all_step_data: list[dict] = []
    all_timeline_steps: list[dict] = []
    all_decomposed_rewards: list[dict] = []
    all_friendly_health_data: list[dict] = []
    all_enemy_health_data: list[dict] = []
    all_unit_actions: list[dict] = []
    all_unit_lifetimes: list[dict] = []
    all_environment_rewards: list[list[float]] = []
    all_value_estimates: list[dict] = []
    all_unit_navigation_rewards: list[dict] = []
    all_unit_combat_rewards: list[dict] = []

    group_by_episode = len(episode_pks) > 1
    for episode_pk in episode_pks:
        insight_results = _episode_insights(episode_pk, group_by_episode)
        all_episode_details.append(insight_results['episode_details'])
        all_action_v_reward[f'Episode {episode_pk}'] = insight_results['action_v_reward']
        all_reward_histogram.append(insight_results['reward_histogram'])
        all_action_v_frequency[f'Episode {episode_pk}'] = insight_results['action_v_frequency']
        all_rewards_over_time.append(insight_results['rewards_over_time'])
        all_decomposed_rewards.append(insight_results['decomposed_rewards'])
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

    # Get unique agents across all episodes
    unique_agents = insight_results['unique_agents']

    # Run clustering task for both observations and action distributions
    obs_clustering, agent_outs_clustering = cluster_episodes_all_features(
        episode_pks,
        umap_n_neighbors=30,
        umap_min_dist=0.5,
        umap_n_components=20,
        pca_n_components=2,
        kmeans_n_clusters=10,
        feature_dimensions='episode time agent',
        seed=42,
    )

    data = {
        'all_episode_details': all_episode_details,
        'parsed_data': {
            'action_v_reward': all_action_v_reward,
            'reward_histogram': all_reward_histogram,
            'action_v_frequency': all_action_v_frequency,
            'rewards_over_time': all_rewards_over_time,
            'decomposed_rewards': all_decomposed_rewards,
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
        },
        'clustering': {
            'obs': {
                'all_manifolds_x': obs_clustering['all_manifolds'][:, 0].tolist(),
                'all_manifolds_y': obs_clustering['all_manifolds'][:, 1].tolist(),
                'all_clusters': obs_clustering['all_clusters'].tolist(),
                'episode_manifolds': obs_clustering['episode_manifolds'].tolist(),
                'episode_clusters': obs_clustering['episode_clusters'][0].T.tolist(),
            },
            'agent_outs': {
                'all_manifolds_x': agent_outs_clustering['all_manifolds'][:, 0].tolist(),
                'all_manifolds_y': agent_outs_clustering['all_manifolds'][:, 1].tolist(),
                'all_clusters': agent_outs_clustering['all_clusters'].tolist(),
                'episode_manifolds': agent_outs_clustering['episode_manifolds'].tolist(),
                'episode_clusters': agent_outs_clustering['episode_clusters'][0].T.tolist(),
            },
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
