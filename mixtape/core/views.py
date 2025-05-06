from collections import defaultdict
from itertools import accumulate
from typing import Any

from django.db.models import Subquery, Sum
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render
import einops
import numpy as np
from sklearn import cluster, decomposition, pipeline, preprocessing
import umap.umap_ as umap

from mixtape.core.models.episode import Episode
from mixtape.core.models.step import Step
from mixtape.core.ray_utils.utility_functions import get_environment_mapping


def _fetch_episode_observations(episode_id: int) -> dict:
    n_timesteps = 0
    n_agents = 0
    unique_agents = None
    agent_to_idx = None
    steps = Step.objects.prefetch_related('agent_steps').filter(episode_id=episode_id)
    if not steps.exists():
        return {'obs': np.zeros((0, 0, 0))}

    n_timesteps = len(steps)
    if unique_agents is None:
        unique_agents = steps.distinct('agent_steps__agent').exclude(agent_steps__agent=None)
        agent_names = unique_agents.values_list('agent_steps__agent', flat=True)
        agent_to_idx = {agent: idx for idx, agent in enumerate(agent_names)}
    n_agents = len(unique_agents)

    # Get observation space shape from first step
    first_step = steps.first()
    if first_step is None:
        return {'obs': np.zeros((0, 0, 0))}

    first_agent_step = first_step.agent_steps.first()
    if first_agent_step is None:
        return {'obs': np.zeros((0, 0, 0))}

    first_obs = np.array(first_agent_step.observation_space).flatten()
    obs_shape = first_obs.shape

    obs = np.zeros((n_timesteps, n_agents, *obs_shape))

    # Fill in the actual data
    for step in steps:
        step_idx = step.number
        for agent_step in step.agent_steps.all():
            agent_idx = agent_to_idx[agent_step.agent]
            obs[step_idx, agent_idx] = np.array(agent_step.observation_space).flatten()

    return {'obs': obs}


def _cluster_episode_by_feature(
    episode: dict,
    feature_name: str,
    manifold_pipeline: pipeline.Pipeline,
    cluster_pipeline: cluster.KMeans,
    dimension_keys: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episode_features = np.stack([episode[feature_name]])

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


def cluster_episode(
    episode_id: int,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.5,
    umap_n_components: int = 20,
    pca_n_components: int = 2,
    kmeans_n_clusters: int = 10,
    feature_name: str = 'obs',
    feature_dimensions: str = 'episode time agent',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episode = _fetch_episode_observations(episode_id)

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
                ),
            ),
            ('decompose', decomposition.PCA(pca_n_components)),
        ]
    )
    cluster_pipeline = cluster.KMeans(kmeans_n_clusters)

    return _cluster_episode_by_feature(
        episode,
        feature_name,
        manifold_pipeline,
        cluster_pipeline,
        dimension_keys=feature_dimensions,
    )


def insights(request: HttpRequest, episode_pk: int) -> HttpResponse:
    # Prefetch all related data in a single query
    episode = get_object_or_404(
        Episode.objects.select_related('inference__checkpoint__training').prefetch_related(
            'steps', 'steps__agent_steps'
        ),
        pk=episode_pk,
    )

    # Run clustering task for this episode
    episode_clusters, episode_manifolds, all_clusters, all_manifolds = cluster_episode(
        episode_pk,
        umap_n_neighbors=30,
        umap_min_dist=0.5,
        umap_n_components=20,
        pca_n_components=2,
        kmeans_n_clusters=10,
        feature_name='obs',
        feature_dimensions='episode time agent',
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
        # clustering results
        'clustering': {
            'all_manifolds_x': all_manifolds[:, 0].tolist(),
            'all_manifolds_y': all_manifolds[:, 1].tolist(),
            'all_clusters': all_clusters.tolist(),
            'episode_manifolds': episode_manifolds.tolist(),
            'episode_clusters': episode_clusters[0].T.tolist(),
        },
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
