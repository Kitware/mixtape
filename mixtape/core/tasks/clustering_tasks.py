"""Tasks for computing clustering results asynchronously."""

from datetime import timedelta
import hashlib
import json
import logging
from typing import Any, Optional

from celery import shared_task
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.db import transaction
from django.db.models.query import QuerySet
from django.utils import timezone

from mixtape.core.analysis.clustering import (
    DEFAULT_CLUSTERING_PARAMS,
    cluster_episodes_all_features,
)
from mixtape.core.analysis.constants import DEFAULT_FEATURE_TYPES
from mixtape.core.models import ClusteringResult, Episode
from mixtape.core.ray_utils.json_encoder import CustomJSONEncoder

logger = logging.getLogger(__name__)
CLUSTERING_LOGIC_VERSION = '1'


def _normalize_params(params: dict | None) -> dict:
    if params is None:
        params = DEFAULT_CLUSTERING_PARAMS
    # Dicts preserve insertion order, so two dicts with the same keys/values but built in different
    # orders can serialize differently. Sort keys to ensure stable hashing.
    return {k: params[k] for k in sorted(params.keys())}


def make_multi_key(episode_ids: list[int], params: Optional[dict] = None) -> str:
    """Deterministic, order-invariant key for multi-episode clustering artifacts."""
    norm_params = _normalize_params(params)
    sorted_ids = sorted(set(int(e) for e in episode_ids))
    payload = {
        'version': CLUSTERING_LOGIC_VERSION,
        'episode_ids': sorted_ids,
        'params': norm_params,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest


@shared_task(bind=True)
def compute_episode_clustering(
    self: Any, episode_ids: list[int], feature_types: Optional[list[str]] = None, **kwargs
) -> dict[str, Any]:
    """Compute clustering results for the specified episodes.

    Args:
        episode_ids: List of episode IDs to cluster
        feature_types: List of feature types to compute. Currently defaults
                       to ['observations', 'agent_outputs'].
        **kwargs: Any additional lustering parameters

    Returns:
        Dict with task status and clustering result IDs
    """
    params = {**DEFAULT_CLUSTERING_PARAMS, **kwargs}

    # Default to all feature types if not specified
    if feature_types is None:
        feature_types = DEFAULT_FEATURE_TYPES

    # In case of race conditions, confirm that all requested episodes actually exist
    existing_episodes: QuerySet = Episode.objects.filter(id__in=episode_ids)
    if existing_episodes.count() != len(episode_ids):
        missing = set(episode_ids) - set(existing_episodes.values_list('id', flat=True))
        raise ValueError(f'Episodes not found: {missing}')

    # FIXME: Current clustering logic only handles one episode at a time
    # TODO: Decide how we want to handle multi-episode clustering
    if len(episode_ids) != 1:
        raise ValueError(
            f'Multi-episode clustering not yet supported, got {len(episode_ids)} episodes'
        )

    episode = existing_episodes.first()
    if episode is None:
        raise ValueError(f'Episodes not found: {episode_ids}')
    created_results: list[int] = []
    try:
        obs_results, agent_outs_results = cluster_episodes_all_features(
            episode_ids=episode_ids, **params
        )

        obs_results_json = json.loads(json.dumps(obs_results, cls=CustomJSONEncoder))
        agent_outs_results_json = json.loads(json.dumps(agent_outs_results, cls=CustomJSONEncoder))

        for feature_type in feature_types:
            existing_result = ClusteringResult.get_for_episode_and_type(
                episode_id=episode.id, feature_type=feature_type, parameters=params
            )

            if existing_result:
                created_results.append(existing_result.id)
                continue

            if feature_type == 'observations':
                results_data = obs_results_json
            elif feature_type == 'agent_outputs':
                results_data = agent_outs_results_json
            else:
                continue

            with transaction.atomic():
                clustering_result = ClusteringResult.objects.create(
                    episode=episode,
                    feature_types=[feature_type],
                    parameters=params,
                    results=results_data,
                    status=ClusteringResult.Status.SUCCESS,
                    error_message=None,
                )
                created_results.append(clustering_result.id)

        return {
            'status': 'success',
            'clustering_result_ids': created_results,
            'episode_ids': episode_ids,
            'feature_types': feature_types,
        }
    except Exception as exc:
        logger.exception('Clustering computation failed for episodes %s', episode_ids)
        raise self.retry(
            exc=exc,  # pass along original exception for logging
            countdown=60,  # wait 60 seconds before retry
            max_retries=3,  # try up to 3 times total
        )


@shared_task
def compute_single_episode_clustering(episode_id: int) -> dict[str, Any]:
    """Compute clustering for a single episode using default parameters.

    Args:
        episode_id: ID of the episode to cluster

    Returns:
        Dict with task status and clustering result ID
    """
    async_result = compute_episode_clustering.delay([episode_id])
    return {
        'status': 'queued',
        'parent_task_id': async_result.id,
        'episode_id': episode_id,
    }


@shared_task(bind=True)
def compute_multi_episode_clustering(
    self: Any, episode_ids: list[int], params: Optional[dict] = None, key: Optional[str] = None
) -> dict[str, Any]:
    """Compute clustering for multiple episodes and store artifact in default storage.

    Stores JSON at clustering-temp/<key>.json with a minimal structure matching template needs.
    """
    try:
        norm_params = _normalize_params(params)
        artifact_key = key or make_multi_key(episode_ids, norm_params)
        path = f'clustering-temp/{artifact_key}.json'

        # Return with results if it already exists
        if default_storage.exists(path):
            return {'status': 'exists', 'key': artifact_key}

        obs_results, agent_outs_results = cluster_episodes_all_features(
            episode_ids=episode_ids, **norm_params
        )

        def serialize(results: dict) -> dict:
            # Build per-episode clusters as [episodes][steps][agents]
            clusters = results['episode_clusters']
            try:
                n_eps = clusters.shape[0]
                ep_clusters = [clusters[i].T.tolist() for i in range(n_eps)]
            except Exception:
                try:
                    ep_clusters = [list(map(list, zip(*ep))) for ep in clusters]
                except Exception:
                    ep_clusters = clusters.tolist() if hasattr(clusters, 'tolist') else clusters

            return {
                'all_manifolds_x': results['all_manifolds'][:, 0].tolist(),
                'all_manifolds_y': results['all_manifolds'][:, 1].tolist(),
                'all_clusters': results['all_clusters'].tolist(),
                'episode_manifolds': results['episode_manifolds'].tolist(),
                'episode_clusters': ep_clusters,
            }

        artifact = {
            'obs': serialize(obs_results),
            'agent_outs': serialize(agent_outs_results),
        }

        content = ContentFile(json.dumps(artifact).encode('utf-8'))
        default_storage.save(path, content)
        return {'status': 'success', 'key': artifact_key}
    except Exception as exc:
        logger.exception('Multi-episode clustering failed for %s: %s', episode_ids, exc)
        raise self.retry(exc=exc, countdown=60, max_retries=3)


@shared_task
def cleanup_clustering_temp(ttl_days: int = 7) -> dict[str, int]:
    """Delete clustering-temp/* artifacts older than ttl_days from default storage."""
    base_prefix = 'clustering-temp'
    deleted = 0
    checked = 0
    cutoff = timezone.now() - timedelta(days=int(ttl_days))

    def _walk(prefix: str) -> list[str]:
        # Collect all file paths under prefix
        files_acc: list[str] = []
        try:
            dirs, files = default_storage.listdir(prefix)
        except Exception as exc:
            logger.debug('Failed to list default_storage for prefix %s: %s', prefix, exc)
            return files_acc
        for f in files:
            files_acc.append(f"{prefix.rstrip('/')}/{f}")
        for d in dirs:
            files_acc.extend(_walk(f"{prefix.rstrip('/')}/{d}"))
        return files_acc

    try:
        all_files = _walk(base_prefix)
        for path in all_files:
            checked += 1
            try:
                mod_time = default_storage.get_modified_time(path)
                if mod_time < cutoff:
                    default_storage.delete(path)
                    deleted += 1
            except Exception as exc:
                # Ignore individual file errors but log at debug for traceability
                logger.debug('Skipping path during cleanup due to error: %s (path=%s)', exc, path)
                continue
    except Exception as exc:
        logger.exception('cleanup_clustering_temp failed: %s', exc)
    return {'checked': checked, 'deleted': deleted}
