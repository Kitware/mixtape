import json

from django.utils.safestring import mark_safe
import numpy as np


def format_float(value, decimal_places=2):
    """Format a float value to the specified number of decimal places."""
    if isinstance(value, float):
        return round(value, decimal_places)
    return value


def format_list(value, max_size=10):
    """
    Format a list value.

    If the list size is larger than max_size, show its shape instead of raw data.
    For lists of floats, round each value to 2 decimal places.
    """
    if not isinstance(value, list):
        return value

    try:
        # Convert to numpy array to get shape
        arr = np.array(value)
        total_size = np.prod(arr.shape)

        # If list is too large, just show its shape
        if total_size > max_size:
            return f'array {arr.shape}'

        # For smaller lists, format each element
        if arr.dtype.kind == 'f':  # If array contains floats
            return [format_float(v) for v in value]
        return value
    except Exception:
        # If conversion to numpy array fails, return as is
        if len(value) > max_size:
            return f'list[{len(value)}]'
        return value


def format_dict(value, max_size=10):
    """
    Format a dictionary value.

    For dict values that are floats, round to 2 decimal places.
    For dict values that are lists, apply list formatting rules.
    """
    if not isinstance(value, dict):
        return value

    result = {}
    for k, v in value.items():
        if isinstance(v, float):
            result[k] = format_float(v)
        elif isinstance(v, list):
            result[k] = format_list(v, max_size)
        elif isinstance(v, dict):
            result[k] = format_dict(v, max_size)
        else:
            result[k] = v
    return result


def format_value(value, max_size=10):
    """
    Format any value.

    Format any value according to the rules:
    - Round floats to 2 decimal places
    - Lists larger than max_size show shape instead of raw data
    - Format dictionaries with the same rules applied to their values
    """
    if isinstance(value, float):
        return format_float(value)
    elif isinstance(value, list):
        return format_list(value, max_size)
    elif isinstance(value, dict):
        return format_dict(value, max_size)
    return value


def pretty_format_dict(value, max_size=10):
    """
    Format a dictionary as pretty-printed HTML.

    Apply all formatting rules to the dictionary values.
    """
    if not value:
        return '-'

    try:
        # Apply formatting rules to the dictionary
        formatted_dict = format_dict(value, max_size)

        # Convert to pretty JSON
        formatted_json = json.dumps(formatted_dict, indent=2, sort_keys=True)

        # Convert to HTML
        formatted_html = formatted_json.replace(' ', '&nbsp;').replace('\n', '<br>')
        return mark_safe(f'<pre style="margin: 0; font-size: 0.8em;">{formatted_html}</pre>')
    except Exception as e:
        return f'Error: {str(e)}'


def format_observation_space(observation_space):
    """
    Format an observation space value.

    For dictionaries, show the shape or type of each value.
    For lists, show the shape.
    """
    if not observation_space:
        return '-'

    try:
        if isinstance(observation_space, dict):
            shapes = {}
            for key, value in observation_space.items():
                if isinstance(value, list):
                    try:
                        shapes[key] = f'array {np.array(value).shape}'
                    except Exception:
                        shapes[key] = f'{type(value).__name__}'
                else:
                    shapes[key] = f'{type(value).__name__}'
            return pretty_format_dict(shapes)
        elif isinstance(observation_space, list):
            try:
                return f'array {np.array(observation_space).shape}'
            except Exception:
                return f'list[{len(observation_space)}]'
        else:
            return f'{type(observation_space).__name__}'
    except Exception as e:
        return f'Error: {str(e)}'
