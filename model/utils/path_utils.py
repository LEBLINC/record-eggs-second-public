import os
import sys


def get_app_root() -> str:
    """Return project root or PyInstaller temp root."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
        if isinstance(base, str) and base:
            return base
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def resource_path(*parts: str) -> str:
    """Build an absolute path to bundled resources."""
    if len(parts) == 1:
        path = parts[0]
        if isinstance(path, str) and os.path.isabs(path):
            return path
        return os.path.join(get_app_root(), str(path))
    return os.path.join(get_app_root(), *[str(p) for p in parts])


def resolve_resource_path(path: str, fallback_rel: str) -> str:
    """Resolve a resource path; fall back when missing."""
    if not isinstance(path, str) or not path:
        return resource_path(fallback_rel)
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        return resource_path(fallback_rel)
    return resource_path(path)
