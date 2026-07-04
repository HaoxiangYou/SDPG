# Lazy import to avoid initializing Warp when importing base_env
# Import is done lazily via __getattr__

import importlib
import sys
from types import ModuleType

# Track which modules are being imported to avoid recursion
_importing = set()


def __getattr__(name: str) -> ModuleType:
    """Lazy import of environment modules."""
    module_name = f"{__name__}.{name}"

    # Check if already imported
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Check if we're already importing this module (avoid recursion)
    if name in _importing:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in ("genesis_env", "mujoco_env"):
        _importing.add(name)
        try:
            # Use importlib to import the submodule directly
            # This bypasses __getattr__ by importing the module object directly
            module = importlib.import_module(module_name)
            return module
        finally:
            _importing.discard(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
