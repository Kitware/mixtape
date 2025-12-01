"""Clustering computation utilities for episode analysis."""

import os

from django.http import Http404
import einops
import numpy as np
from sklearn import cluster, decomposition, pipeline, preprocessing
import umap.umap_ as umap

from mixtape.core.analysis.constants import DEFAULT_CLUSTERING_DIMENSIONS
from mixtape.core.models.step import Step

# Default clustering parameters used across the application
DEFAULT_CLUSTERING_PARAMS = {
    'umap_n_neighbors': 30,
    'umap_min_dist': 0.5,
    'umap_n_components': 20,
    'pca_n_components': 2,
    'kmeans_n_clusters': 10,
    'feature_dimensions': DEFAULT_CLUSTERING_DIMENSIONS,
    'seed': 42,
}


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
    feature_dimensions: str = DEFAULT_CLUSTERING_DIMENSIONS,
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
