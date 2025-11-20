from collections import defaultdict
from itertools import accumulate
import json
import logging
import os

from django.db import transaction
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render
import numpy as np
from sklearn import cluster, decomposition, pipeline, preprocessing
import umap.umap_ as umap

from mixtape.core.analysis.clustering import (
    DEFAULT_CLUSTERING_PARAMS,
    cluster_episodes_all_features,
)
from mixtape.core.analysis.clustering.algorithms import (
    _cluster_episodes_by_feature,
    _fetch_all_episode_features,
)
from mixtape.core.models import ClusteringResult, Episode
from mixtape.core.ray_utils.json_encoder import CustomJSONEncoder
from mixtape.core.ray_utils.utility_functions import get_environment_mapping

logger = logging.getLogger(__name__)


def get_or_compute_clustering_results(episode_ids: list[int], **kwargs) -> tuple[dict, dict]:
    """Get pre-computed clustering results or fallback to runtime computation.

    Args:
        episode_ids: List of episode IDs to cluster
        **kwargs: Any additional clustering parameters to set or override

    Returns:
        Tuple of dictionaries keyed by feature type
    """
    params = {**DEFAULT_CLUSTERING_PARAMS, **kwargs}
    if len(episode_ids) == 1:
        episode_id = episode_ids[0]
        obs_clustering = ClusteringResult.get_for_episode_and_type(
            episode_id=episode_id, feature_type='observations', parameters=params
        )
        agent_outs_clustering = ClusteringResult.get_for_episode_and_type(
            episode_id=episode_id, feature_type='agent_outputs', parameters=params
        )

        # Use any pre-computed results; compute missing types at runtime
        if obs_clustering and agent_outs_clustering:
            return obs_clustering.results, agent_outs_clustering.results

        # Compute only missing features
        computed_obs: dict | None = None
        computed_agent: dict | None = None

        def _compute_feature(feature_key: str) -> dict:
            avail_cpus = (os.cpu_count() or 0) - 1
            manifold_pipeline = pipeline.Pipeline(
                [
                    ('scale', preprocessing.StandardScaler()),
                    (
                        'manifold',
                        umap.UMAP(
                            n_neighbors=params['umap_n_neighbors'],
                            min_dist=params['umap_min_dist'],
                            n_components=params['umap_n_components'],
                            densmap=False,
                            n_jobs=min(8, max(1, avail_cpus)),
                        ),
                    ),
                    ('decompose', decomposition.PCA(params['pca_n_components'])),
                ]
            )
            cluster_pipeline = cluster.KMeans(
                params['kmeans_n_clusters'], random_state=params['seed']
            )

            feats = _fetch_all_episode_features(episode_ids)
            if feature_key == 'observations':
                episodes = [{'obs': feat['obs']} for feat in feats if 'obs' in feat]
                name = 'obs'
            else:
                episodes = [
                    {'agent_outs': feat['agent_outs']} for feat in feats if 'agent_outs' in feat
                ]
                name = 'agent_outs'

            ep_clusters, ep_manifolds, all_clusters, all_manifolds = _cluster_episodes_by_feature(
                episodes,
                name,
                manifold_pipeline,
                cluster_pipeline,
                dimension_keys=params['feature_dimensions'],
            )
            return {
                'episode_clusters': ep_clusters,
                'episode_manifolds': ep_manifolds,
                'all_clusters': all_clusters,
                'all_manifolds': all_manifolds,
            }

        if not obs_clustering:
            computed_obs = _compute_feature('observations')
            try:
                with transaction.atomic():
                    ClusteringResult.objects.create(
                        episode_id=episode_id,
                        feature_types=['observations'],
                        parameters=params,
                        results=json.loads(json.dumps(computed_obs, cls=CustomJSONEncoder)),
                        status=ClusteringResult.Status.SUCCESS,
                        error_message=None,
                    )
            except Exception as exc:
                logger.exception(
                    'Failed to persist observations clustering for episode_id=%s: %s',
                    episode_id,
                    exc,
                )

        if not agent_outs_clustering:
            computed_agent = _compute_feature('agent_outputs')
            try:
                with transaction.atomic():
                    ClusteringResult.objects.create(
                        episode_id=episode_id,
                        feature_types=['agent_outputs'],
                        parameters=params,
                        results=json.loads(json.dumps(computed_agent, cls=CustomJSONEncoder)),
                        status=ClusteringResult.Status.SUCCESS,
                        error_message=None,
                    )
            except Exception as exc:
                logger.exception(
                    'Failed to persist agent_outputs clustering for episode_id=%s: %s',
                    episode_id,
                    exc,
                )

        # Prefer cached where available
        return (
            obs_clustering.results if obs_clustering else (computed_obs or {}),
            agent_outs_clustering.results if agent_outs_clustering else (computed_agent or {}),
        )

    # If no pre-computed results found or multi-episode, compute at runtime
    return cluster_episodes_all_features(episode_ids=episode_ids, **params)


def _episode_insights(episode_pk: int, group_by_episode: bool = False) -> dict:
    # Prefetch all related data in a single query
    episode = get_object_or_404(
        Episode.objects.select_related('inference__checkpoint__training').prefetch_related(
            'steps', 'steps__agent_steps'
        ),
        pk=episode_pk,
    )

    # Cache frequently used data to avoid repeated queries
    all_steps = episode.steps.all()
    steps_agent_steps = {step: list(step.agent_steps.all()) for step in all_steps}
    env_name = episode.inference.checkpoint.training.environment
    training = episode.inference.checkpoint.training
    action_map = get_environment_mapping(env_name)

    # Prepare step data
    step_data = {}
    for step in all_steps:
        agent_steps_data = []
        for agent_step in steps_agent_steps[step]:
            action_string = action_map.get(f'{int(agent_step.action)}', f'{agent_step.action}')
            agent_steps_data.append(
                {
                    'agent': agent_step.agent,
                    'action': action_string,
                    'total_reward': agent_step.total_reward,
                }
            )
        step_data[step.number] = {
            'image_url': step.image.url if step.image else None,
            'agent_steps': agent_steps_data,
        }

    reward_histogram = []
    for step in all_steps:
        for agent_step in steps_agent_steps[step]:
            reward_histogram.append(agent_step.total_reward)

    action_v_reward: defaultdict[str, float] | defaultdict[str, defaultdict[str, float]] = (
        defaultdict(float if group_by_episode else lambda: defaultdict(float))  # type: ignore
    )
    action_v_frequency: defaultdict[str, int] | defaultdict[str, defaultdict[str, int]] = (
        defaultdict(int if group_by_episode else lambda: defaultdict(int))  # type: ignore
    )

    unique_agents = set()
    reward_mapping = training.reward_mapping or []
    decomposed_rewards: dict[str, list[float]] = {v: [] for v in reward_mapping}

    for step in all_steps:
        step_decomposed_rewards = {v: 0.0 for v in reward_mapping}
        for agent_step in steps_agent_steps[step]:
            unique_agents.add(agent_step.agent)
            action = action_map.get(f'{int(agent_step.action)}', f'{agent_step.action}')
            key = action if group_by_episode else agent_step.agent
            if group_by_episode:
                action_v_reward[action] += agent_step.total_reward  # type: ignore
                action_v_frequency[action] += 1  # type: ignore
            else:
                action_v_reward[key][action] += agent_step.total_reward  # type: ignore
                action_v_frequency[key][action] += 1  # type: ignore
            for i, reward_name in enumerate(reward_mapping):
                step_decomposed_rewards[reward_name] += agent_step.rewards[i]
        for reward_name in reward_mapping:
            decomposed_rewards[reward_name].append(step_decomposed_rewards[reward_name])

    # Calculate cumulative decomposed rewards
    cumulative_decomposed_rewards = {}
    for reward_name, rewards_list in decomposed_rewards.items():
        cumulative_decomposed_rewards[reward_name] = list(accumulate(rewards_list))

    # Populate the timeline
    key_steps = sorted(all_steps, key=lambda s: s.number)

    # In Python since total_reward is a property
    key_steps_values = []
    for step in key_steps:
        key_steps_values.append(
            {
                'number': step.number,
                'total_rewards': sum(
                    agent_step.total_reward for agent_step in steps_agent_steps[step]
                ),
            }
        )

    # list of cumulative rewards over time
    rewards_over_time = list(accumulate(ks['total_rewards'] for ks in key_steps_values))
    episode_total_rewards = sum(ks['total_rewards'] for ks in key_steps_values)
    # TODO: Revist this. This exists more as a placeholder, timeline
    #       should represent points of interest with more meaning.
    # Get top 40 steps by total rewards, ordered by step number
    timeline_steps = sorted(
        [step for step in key_steps_values if step['total_rewards'] > 0],
        key=lambda x: x['total_rewards'],
        reverse=True,
    )[:40]
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
        'episode_total_rewards': episode_total_rewards,
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
    all_episode_total_rewards: dict[int, int | float] = {}

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
        all_episode_total_rewards[episode_pk] = insight_results['episode_total_rewards']

    # Get unique agents across all episodes
    unique_agents = insight_results['unique_agents']

    # Single-episode: compute or use precomputed; no multi-episode artifact logic in this commit
    obs_clustering, agent_outs_clustering = get_or_compute_clustering_results(episode_pks)
    clustering_results = [obs_clustering, agent_outs_clustering]
    array_fields = ['all_manifolds', 'all_clusters', 'episode_manifolds', 'episode_clusters']
    for clustering in clustering_results:
        for field in array_fields:
            if isinstance(clustering[field], list):
                clustering[field] = np.array(clustering[field])

    # Get episode total rewards and average rewards (ordered by episode_pks)
    episode_total_rewards = [
        all_episode_total_rewards[pk] for pk in episode_pks if pk in all_episode_total_rewards
    ]

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
            'step_data': all_step_data,
            'timeline_key_steps': all_timeline_steps,
            'episode_total_rewards': episode_total_rewards,
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
    }

    return render(
        request,
        'core/insights/insights.html',
        data,
    )
    # No multi-episode endpoints in this commit


def home_page(request: HttpRequest) -> HttpResponse:
    episodes = Episode.objects.select_related('inference__checkpoint__training').all()
    return render(request, 'core/home/home.html', {'episodes': episodes})
