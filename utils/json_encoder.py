"""
JSON Encoder utilities to handle numpy types and other non-serializable objects
"""

import json
import numpy as np
from datetime import datetime
from typing import Any

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):  # Handle other numpy-like objects
            return obj.tolist()
        return super().default(obj)

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert an object to be JSON serializable
    """
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    elif hasattr(obj, 'tolist'):  # Handle other numpy-like objects
        try:
            return obj.tolist()
        except (ValueError, AttributeError):
            pass
    
    # Try to convert to string as last resort for unknown types
    try:
        # Test if it's already JSON serializable
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize an object to JSON string
    """
    sanitized_obj = sanitize_for_json(obj)
    return json.dumps(sanitized_obj, cls=NumpyJSONEncoder, **kwargs)