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
