from collections import defaultdict
from itertools import accumulate
import json
import logging
import os

from django.core.files.storage import default_storage
from django.db import transaction
from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
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
from mixtape.core.tasks.clustering_tasks import (
    compute_multi_episode_clustering,
    compute_single_episode_clustering,
    make_multi_key,
)

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

    # Get clustering results if available; otherwise enqueue async compute and render placeholders
    clustering_available = False
    clustering_task_id = None
    obs_clustering = None
    agent_outs_clustering = None
    artifact_clustering = None
    if len(episode_pks) == 1:
        params = {**DEFAULT_CLUSTERING_PARAMS}
        obs_result = ClusteringResult.get_for_episode_and_type(
            episode_id=episode_pks[0], feature_type='observations', parameters=params
        )
        agent_result = ClusteringResult.get_for_episode_and_type(
            episode_id=episode_pks[0], feature_type='agent_outputs', parameters=params
        )
        if obs_result and agent_result:
            obs_clustering = obs_result.results
            agent_outs_clustering = agent_result.results
            clustering_available = True
        else:
            try:
                async_res = compute_single_episode_clustering.delay(episode_pks[0])
                clustering_task_id = async_res.id
            except Exception as exc:
                logger.exception(
                    'Failed to enqueue single-episode clustering for episode_id=%s: %s',
                    episode_pks[0],
                    exc,
                )
                clustering_task_id = None
            # If one of the results is already available, include it now so UI can render partially
            if obs_result and not obs_clustering:
                obs_clustering = obs_result.results
            if agent_result and not agent_outs_clustering:
                agent_outs_clustering = agent_result.results
    else:
        # Multi-episode: prefer async compute with storage artifact; avoid DB storage
        params = {**DEFAULT_CLUSTERING_PARAMS}
        multi_key = make_multi_key(episode_pks, params)
        artifact_path = f'clustering-temp/{multi_key}.json'
        if default_storage.exists(artifact_path):
            try:
                with default_storage.open(artifact_path, 'rb') as fh:
                    artifact_clustering = json.load(fh)
                clustering_available = True
            except Exception as exc:
                logger.exception('Failed to read clustering artifact %s: %s', artifact_path, exc)
                artifact_clustering = None
                clustering_available = False
        else:
            try:
                compute_multi_episode_clustering.delay(episode_pks, params, multi_key)
                clustering_task_id = None
            except Exception as exc:
                logger.exception(
                    'Failed to enqueue multi-episode clustering for %s: %s', episode_pks, exc
                )
                clustering_task_id = None

    if clustering_available and obs_clustering and agent_outs_clustering:
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

    # Helper to serialize clustering results regardless of list/ndarray types
    def _serialize_results(results: dict) -> dict:
        import numpy as _np

        r = results
        am = (
            _np.array(r['all_manifolds'])
            if isinstance(r.get('all_manifolds'), list)
            else r['all_manifolds']
        )
        ac = (
            _np.array(r['all_clusters'])
            if isinstance(r.get('all_clusters'), list)
            else r['all_clusters']
        )
        em = (
            _np.array(r['episode_manifolds'])
            if isinstance(r.get('episode_manifolds'), list)
            else r['episode_manifolds']
        )
        ec = (
            _np.array(r['episode_clusters'])
            if isinstance(r.get('episode_clusters'), list)
            else r['episode_clusters']
        )
        return {
            'all_manifolds_x': am[:, 0].tolist(),
            'all_manifolds_y': am[:, 1].tolist(),
            'all_clusters': ac.tolist(),
            'episode_manifolds': em.tolist(),
            'episode_clusters': ec[0].T.tolist(),
        }

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
        'clustering': (
            artifact_clustering
            if artifact_clustering is not None
            else (
                (
                    {
                        **({'obs': _serialize_results(obs_clustering)} if obs_clustering else {}),
                        **(
                            {'agent_outs': _serialize_results(agent_outs_clustering)}
                            if agent_outs_clustering
                            else {}
                        ),
                    }
                    if (obs_clustering or agent_outs_clustering)
                    else None
                )
            )
        ),
        'has_reward_mapping': any(
            episode.inference.checkpoint.training.reward_mapping for episode in all_episode_details
        ),
        'clustering_available': clustering_available,
        'clustering_task_id': clustering_task_id,
        'multi_key': (
            make_multi_key(episode_pks, {**DEFAULT_CLUSTERING_PARAMS})
            if len(episode_pks) > 1
            else ''
        ),
    }

    return render(
        request,
        'core/insights/insights.html',
        data,
    )
    # End insights


def clustering_status(request: HttpRequest) -> HttpResponse:
    """Return JSON indicating whether clustering results are available for an episode.

    Supports multi-episode by key or by list; and single-episode by id.
    """
    # Multi-episode by key
    multi_key = request.GET.get('multi_key')
    if multi_key:
        path = f'clustering-temp/{multi_key}.json'
        return JsonResponse({'available': default_storage.exists(path)})

    # Optional local override for single-episode ID derived from episode_ids
    episode_id_override: str | None = None

    # Multi-episode by episode_ids
    episode_ids = request.GET.get('episode_ids')
    if episode_ids:
        try:
            ids = [int(x) for x in episode_ids.split(',') if x.strip()]
        except ValueError:
            return JsonResponse({'error': 'invalid episode_ids'}, status=400)
        if len(ids) > 1:
            key = make_multi_key(ids, {**DEFAULT_CLUSTERING_PARAMS})
            path = f'clustering-temp/{key}.json'
            return JsonResponse({'available': default_storage.exists(path)})
        if len(ids) == 1:
            episode_id_override = str(ids[0])

    episode_id = episode_id_override or request.GET.get('episode_id')
    if not episode_id:
        return JsonResponse({'error': 'missing episode_id'}, status=400)
    try:
        eid = int(episode_id)
    except ValueError:
        return JsonResponse({'error': 'invalid episode_id'}, status=400)

    params = {**DEFAULT_CLUSTERING_PARAMS}
    obs_result = ClusteringResult.get_for_episode_and_type(
        episode_id=eid, feature_type='observations', parameters=params
    )
    agent_result = ClusteringResult.get_for_episode_and_type(
        episode_id=eid, feature_type='agent_outputs', parameters=params
    )
    available = bool(obs_result and agent_result)
    return JsonResponse(
        {
            'available': available,
            'obs_available': bool(obs_result),
            'agent_outs_available': bool(agent_result),
        }
    )


def clustering_result(request: HttpRequest) -> HttpResponse:
    """Return clustering JSON for single- or multi-episode.

    Accepts:
    - multi_key=<key> (preferred for multi)
    - episode_ids=1,2,3 (server derives key)
    - episode_id=<id> (single-episode)
    """
    # Multi-episode by key or list
    multi_key = request.GET.get('multi_key')
    episode_ids_param = request.GET.get('episode_ids')
    if multi_key or episode_ids_param:
        if not multi_key:
            try:
                ids = [int(x) for x in (episode_ids_param or '').split(',') if x.strip()]
            except ValueError:
                return JsonResponse({'error': 'invalid episode_ids'}, status=400)
            if len(ids) < 2:
                return JsonResponse({'error': 'need >=2 episode_ids for multi'}, status=400)
            multi_key = make_multi_key(ids, {**DEFAULT_CLUSTERING_PARAMS})
        path = f'clustering-temp/{multi_key}.json'
        if not default_storage.exists(path):
            return JsonResponse({'error': 'not ready'}, status=404)
        with default_storage.open(path, 'rb') as fh:
            try:
                obj = json.load(fh)
            except Exception:
                return JsonResponse({'error': 'corrupt artifact'}, status=500)
        return JsonResponse(obj)

    # Single-episode
    episode_id = request.GET.get('episode_id')
    if not episode_id:
        return JsonResponse({'error': 'missing episode_id or multi_key/episode_ids'}, status=400)
    try:
        eid = int(episode_id)
    except ValueError:
        return JsonResponse({'error': 'invalid episode_id'}, status=400)
    params = {**DEFAULT_CLUSTERING_PARAMS}
    obs_result = ClusteringResult.get_for_episode_and_type(
        episode_id=eid, feature_type='observations', parameters=params
    )
    agent_result = ClusteringResult.get_for_episode_and_type(
        episode_id=eid, feature_type='agent_outputs', parameters=params
    )
    # Allow partial results: return whichever feature types are ready
    if not (obs_result or agent_result):
        return JsonResponse({'error': 'not ready'}, status=404)

    def serialize(results: dict) -> dict:
        import numpy as _np

        r = results
        am = (
            _np.array(r['all_manifolds'])
            if isinstance(r.get('all_manifolds'), list)
            else r['all_manifolds']
        )
        ac = (
            _np.array(r['all_clusters'])
            if isinstance(r.get('all_clusters'), list)
            else r['all_clusters']
        )
        em = (
            _np.array(r['episode_manifolds'])
            if isinstance(r.get('episode_manifolds'), list)
            else r['episode_manifolds']
        )
        ec = (
            _np.array(r['episode_clusters'])
            if isinstance(r.get('episode_clusters'), list)
            else r['episode_clusters']
        )
        return {
            'all_manifolds_x': am[:, 0].tolist(),
            'all_manifolds_y': am[:, 1].tolist(),
            'all_clusters': ac.tolist(),
            'episode_manifolds': em.tolist(),
            'episode_clusters': ec[0].T.tolist(),
        }

    payload: dict[str, dict] = {}
    if obs_result:
        payload['obs'] = serialize(obs_result.results)
    if agent_result:
        payload['agent_outs'] = serialize(agent_result.results)
    return JsonResponse(payload)


def home_page(request: HttpRequest) -> HttpResponse:
    episodes = Episode.objects.select_related('inference__checkpoint__training').all()
    return render(request, 'core/home/home.html', {'episodes': episodes})
