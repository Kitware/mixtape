"""Shared constants for clustering and feature processing."""

# Types of features that can be clustered
FEATURE_TYPES = [
    'observations',
    'agent_outputs',
    'rewards',
    'states',
    'actions',
]

# Dimensions along which clustering can be organized
CLUSTERING_DIMENSIONS = [
    'episode',
    'time',
    'agent',
]

# Default feature types for automatic clustering
DEFAULT_FEATURE_TYPES = ['observations', 'agent_outputs']

# Default clustering dimensions
DEFAULT_CLUSTERING_DIMENSIONS = 'episode time agent'

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
