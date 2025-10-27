"""Tasks for computing clustering results asynchronously."""

import json
import logging
from typing import Any, Optional

from celery import shared_task
from django.db import transaction
from django.db.models.query import QuerySet

from mixtape.core.analysis.clustering import (
    DEFAULT_CLUSTERING_PARAMS,
    cluster_episodes_all_features,
)
from mixtape.core.analysis.constants import DEFAULT_FEATURE_TYPES
from mixtape.core.models import ClusteringResult, Episode
from mixtape.core.ray_utils.json_encoder import CustomJSONEncoder

logger = logging.getLogger(__name__)


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
            exc=exc,        # pass along original exception for logging
            countdown=60,   # wait 60 seconds before retry
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
