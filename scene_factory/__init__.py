"""Photo-driven batch Blender scenes; importing this package does not load Blender."""
from pathlib import Path
import sys

__version__ = '0.3.15'


def _enable_legacy_imports():
    """Bridge standalone Blender scripts while package imports are migrated."""
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
