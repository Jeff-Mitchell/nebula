import os
import time
import logging
from typing import Any, Tuple

import numpy as np

from nebula.config.config import TRAINING_LOGGER

logging_training = logging.getLogger(TRAINING_LOGGER)


def wait_for_file(file_path: str, poll_interval: float = 1.0) -> None:
    """Wait until the given file exists, polling every poll_interval seconds.

    Args:
        file_path: Path to the file to wait for.
        poll_interval: Seconds to wait between checks.
    """
    while not os.path.exists(file_path):
        logging_training.info(f"Waiting for file: {file_path}")
        time.sleep(poll_interval)


def to_numpy(array_like: Any) -> np.ndarray | None:
    """Try to convert tensors/lists/arrays to a numpy array.

    Returns None if conversion is not possible.
    """
    try:
        if isinstance(array_like, np.ndarray):
            return array_like
        # torch tensors (duck-typed): supports detach/cpu/numpy
        if hasattr(array_like, "detach") and hasattr(array_like, "cpu") and hasattr(array_like, "numpy"):
            return array_like.detach().cpu().numpy()
        # Generic sequence
        return np.asarray(array_like)
    except Exception:
        return None


def safe_index_array(arr: Any, indices: np.ndarray) -> np.ndarray | None:
    """Index numpy arrays, torch tensors or Python lists safely.

    Returns None if indexing is not possible.
    """
    try:
        if arr is None:
            return None
        import numpy as _np
        # Numpy array
        if isinstance(arr, _np.ndarray):
            return arr[indices]
        # Torch tensor (duck-typed)
        if hasattr(arr, "detach") and hasattr(arr, "cpu") and hasattr(arr, "numpy"):
            return arr.detach().cpu().numpy()[indices]
        # Python list/tuple (e.g., targets in torchvision)
        if isinstance(arr, (list, tuple)):
            arr_np = _np.asarray(arr)
            return arr_np[indices]
    except Exception:
        return None
    return None


def collect_arrays(ds: Any, indices: np.ndarray) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Collect samples and labels as numpy arrays by iterating a Dataset.

    Tries ds[i] expecting either (x, y) or x and fetches y from ds.targets when available.
    Returns (None, None) on failure or shape inconsistencies.
    """
    try:
        samples = []
        labels = []
        for i in indices.tolist():
            item = ds[i]
            # item can be tuple (x, y) or a single value
            if isinstance(item, tuple) and len(item) >= 2:
                x, y = item[0], item[1]
            else:
                # If targets attribute exists, fetch y from there
                x = item
                y_attr = getattr(ds, "targets", None)
                y = y_attr[i] if y_attr is not None else None
            x_np = to_numpy(x)
            if x_np is None:
                return None, None
            samples.append(x_np)
            labels.append(int(y) if y is not None else 0)
        # Try to stack; if shapes are inconsistent, this will fail
        data_np = np.stack(samples)
        targets_np = np.asarray(labels)
        return data_np, targets_np
    except Exception as e:
        logging_training.exception(f"Failed to collect arrays from dataset: {e}")
        return None, None


def extract_arrays(ds: Any, indices: np.ndarray) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Best-effort extraction of contiguous numpy arrays for data and targets.

    - Tries direct indexing over `ds.data` / `ds.targets` via `safe_index_array`.
    - Falls back to iterating with `collect_arrays` when direct indexing is not possible.
    - Avoids returning object-dtype arrays (returns None for data in that case).
    """
    try:
        data_attr = getattr(ds, "data", None)
        targets_attr = getattr(ds, "targets", None)
        data_np = safe_index_array(data_attr, indices)
        targets_np = safe_index_array(targets_attr, indices)
        if data_np is None or targets_np is None:
            data_np2, targets_np2 = collect_arrays(ds, indices)
            if data_np is None:
                data_np = data_np2
            if targets_np is None:
                targets_np = targets_np2
        if isinstance(data_np, np.ndarray) and data_np.dtype == object:
            data_np = None
        return data_np, targets_np
    except Exception as e:
        logging_training.exception(f"extract_arrays failed: {e}")
        return None, None
