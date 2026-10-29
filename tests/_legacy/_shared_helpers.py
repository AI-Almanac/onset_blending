# Internal helpers for read_spec.py


def _has_path(x, path):
    """Check whether a dotted path exists in a nested dict."""
    parts = path.split(".")
    cur = x
    for p in parts:
        if not isinstance(cur, dict) or p not in cur or cur[p] is None:
            return False
        cur = cur[p]
    return True


def _get_path(x, path):
    """Retrieve a value at a dotted path from a nested dict."""
    parts = path.split(".")
    cur = x
    for p in parts:
        cur = cur[p]
    return cur
