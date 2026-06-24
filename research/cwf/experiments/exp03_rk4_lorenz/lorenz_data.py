"""Re-export of Lorenz data generator from exp02_lorenz.

Avoids duplicating the trajectory generator and oracle code.

Why this is non-trivial: `exp02_lorenz/lorenz_data.py` does relative
imports like `from lorenz_oracle import LorenzOracle`, which only
resolve when exp02_lorenz/ is on sys.path. But exp02_lorenz has no
__init__.py (it predates the package convention), so the relative
imports only work if you `sys.path.insert(0, exp02_lorenz_dir)` AND
load the file by absolute path.

We use `importlib.util.spec_from_file_location` to load the exp02
module under a unique synthetic name — this sidesteps both the
self-import collision (if we did `from lorenz_data import ...` inside
our own `lorenz_data.py`) and the package-resolution issue (we don't
need exp02_lorenz to be a real package).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_EXP02_LORENZ_DATA = Path(__file__).resolve().parents[1] / "exp02_lorenz" / "lorenz_data.py"


def _load_exp02_lorenz_data() -> ModuleType:
    """Load exp02_lorenz.lorenz_data as an isolated module under a synthetic name."""
    exp02_dir = _EXP02_LORENZ_DATA.parent
    # Ensure relative imports inside the loaded module resolve
    if str(exp02_dir) not in sys.path:
        sys.path.insert(0, str(exp02_dir))
    spec = importlib.util.spec_from_file_location(
        "_exp02_lorenz_data_reexport",
        str(_EXP02_LORENZ_DATA),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {_EXP02_LORENZ_DATA}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_exp02 = _load_exp02_lorenz_data()
generate_lorenz_trajectories = _exp02.generate_lorenz_trajectories