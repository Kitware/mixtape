from collections import defaultdict
from itertools import accumulate
from typing import Any

from django.db.models import Subquery, Sum
from django.http import HttpRequest, HttpResponse, Http404
from django.shortcuts import get_object_or_404, render
import einops
import numpy as np
from sklearn import cluster, decomposition, pipeline, preprocessing
import umap.umap_ as umap

from mixtape.core.models.episode import Episode
from mixtape.core.models.step import Step
from mixtape.core.ray_utils.utility_functions import get_environment_mapping


def _fetch_all_episode_observations(episode_ids: list[int]) -> list[dict]:
    """
    Fetch observations for multiple episodes and pad them to match the episode with the most steps.

    Args:
        episode_ids: List of episode IDs to fetch observations for

    Returns:
        List of dictionaries, each containing an 'obs' key with a padded numpy array
    """
    # Fetch all episodes and their steps
    all_steps = Step.objects.prefetch_related('agent_steps').filter(episode_id__in=episode_ids)

    # Group steps by episode
    episode_steps = {}
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
    unique_agents = all_agent_steps.distinct('agent_steps__agent').values_list('agent_steps__agent', flat=True)
    agent_to_idx = {agent: idx for idx, agent in enumerate(unique_agents)}
    n_agents = len(unique_agents)

    # Get observation space shape from first step
    first_step = episode_steps[episode_ids[0]][0]
    first_agent_step = first_step.agent_steps.first()
    first_obs = np.array(first_agent_step.observation_space).flatten()
    obs_shape = first_obs.shape

    results = []

    for episode_id in episode_ids:
        steps = episode_steps[episode_id]

        # Initialize padded array
        obs = np.zeros((max_timesteps, n_agents, *obs_shape))

        # Fill in the actual data
        for step in steps:
            step_idx = step.number
            for agent_step in step.agent_steps.all():
                if agent_step.agent in agent_to_idx:
                    agent_idx = agent_to_idx[agent_step.agent]
                    obs[step_idx, agent_idx] = np.array(agent_step.observation_space).flatten()

        results.append({'obs': obs})

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


def cluster_episodes(
    episode_ids: list[int],
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.5,
    umap_n_components: int = 20,
    pca_n_components: int = 2,
    kmeans_n_clusters: int = 10,
    feature_name: str = 'obs',
    feature_dimensions: str = 'episode time agent',
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episodes = _fetch_all_episode_observations(episode_ids)

    manifold_pipeline = pipeline.Pipeline(
        [
            ('scale', preprocessing.StandardScaler()),
            (
                'manifold',
                umap.UMAP(
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    n_components=umap_n_components,
                    densmap=True,
                    random_state=seed,
                ),
            ),
            ('decompose', decomposition.PCA(pca_n_components)),
        ]
    )
    cluster_pipeline = cluster.KMeans(kmeans_n_clusters, random_state=seed)

    return _cluster_episodes_by_feature(
        episodes,
        feature_name,
        manifold_pipeline,
        cluster_pipeline,
        dimension_keys=feature_dimensions,
    )


def _episode_insights(episode_pk: int, group_by_episode: bool = False) -> HttpResponse:
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
    reward_histogram = [
        a.reward for step in episode.steps.all() for a in step.agent_steps.all()
    ]

    action_map = get_environment_mapping(env_name)
    action_v_reward = defaultdict(float if group_by_episode else lambda: defaultdict(float))
    action_v_frequency = defaultdict(int if group_by_episode else lambda: defaultdict(int))

    unique_agents = set()
    for step in episode.steps.all():
        for agent_step in step.agent_steps.all():
            unique_agents.add(agent_step.agent)
            action = action_map.get(f'{int(agent_step.action)}', f'{agent_step.action}')
            key = action if group_by_episode else agent_step.agent
            if group_by_episode:
                action_v_reward[action] += agent_step.reward
                action_v_frequency[action] += 1
            else:
                action_v_reward[key][action] += agent_step.reward
                action_v_frequency[key][action] += 1

    unique_agents = list(unique_agents)

    # Used to populate the timeline
    key_steps = (
        episode.steps.all()
        .annotate(total_rewards=Sum('agent_steps__reward', default=0))
        .order_by('number')
    )
    # list of cumulative rewards over time
    rewards_over_time = list(accumulate(ks.total_rewards for ks in key_steps))
    # TODO: Revist this. This exists more as a placeholder, timeline
    #       should represent points of interest with more meaning.
    # Get top 40 steps by total rewards, ordered by step number
    timeline_steps = key_steps.filter(
        id__in=Subquery(
            key_steps.filter(total_rewards__gt=0).order_by('-total_rewards').values('id')[:40]
        )
    ).order_by('number')
    timeline_steps_serialized = [{
            'number': step.number,
            'total_rewards': step.total_rewards,
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
        'step_data': step_data,
        'unique_agents': unique_agents,
    }


def insights(request: HttpRequest) -> HttpResponse:
    # Get episode IDs from query parameters
    episode_ids = request.GET.getlist('episode_id')
    if not episode_ids:
        raise Http404("No episode IDs provided")
    try:
        episode_pks = [int(episode_id) for episode_id in episode_ids]
    except ValueError:
        raise Http404("Invalid episode ID format")

    # Process each episode and combine the data
    all_episode_details = []
    all_action_v_reward = defaultdict(lambda: defaultdict(float))
    all_reward_histogram = []
    all_action_v_frequency = defaultdict(lambda: defaultdict(int))
    all_rewards_over_time = []
    all_step_data = []
    all_timeline_steps = []

    group_by_episode = len(episode_pks) > 1
    for episode_pk in episode_pks:
        insight_results = _episode_insights(episode_pk, group_by_episode)
        all_episode_details.append(insight_results['episode_details'])
        all_action_v_reward[f'Episode {episode_pk}'] = insight_results['action_v_reward']
        all_reward_histogram.append(insight_results['reward_histogram'])
        all_action_v_frequency[f'Episode {episode_pk}'] = insight_results['action_v_frequency']
        all_rewards_over_time.append(insight_results['rewards_over_time'])
        all_step_data.append(insight_results['step_data'])
        all_timeline_steps.append(insight_results['timeline_key_steps'])

    # Get unique agents across all episodes
    unique_agents = insight_results['unique_agents']

    # Run clustering task for this episode
    episode_clusters, episode_manifolds, all_clusters, all_manifolds = cluster_episodes(
        episode_pks,
        umap_n_neighbors=30,
        umap_min_dist=0.5,
        umap_n_components=20,
        pca_n_components=2,
        kmeans_n_clusters=10,
        feature_name='obs',
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
            'unique_agents': unique_agents,
            'episode_ids': episode_pks,
            'max_steps': max(len(step_data) for step_data in all_step_data),
            'step_data': all_step_data,
            'timeline_key_steps': all_timeline_steps,
        },
        'clustering': {
            'all_manifolds_x': all_manifolds[:, 0].tolist(),
            'all_manifolds_y': all_manifolds[:, 1].tolist(),
            'all_clusters': all_clusters.tolist(),
            'episode_manifolds': episode_manifolds.tolist(),
            'episode_clusters': episode_clusters[0].T.tolist(),
        }
    }

    return render(
        request,
        'core/insights.html',
        data,
    )


def home_page(request: HttpRequest) -> HttpResponse:
    episodes = Episode.objects.select_related('inference__checkpoint__training').all()
    return render(request, 'core/home.html', {'episodes': episodes})
