from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_stage_module(stage: str):
    """Load one existing numbered RNQI stage as a Python module.

    The project keeps the production stages as standalone scripts. This loader
    lets the package reuse their existing function implementations without
    copying those functions into a second implementation.
    """
    candidates = [
        ROOT / f"{stage}.py",
        *sorted(ROOT.glob(f"{stage}(*).py"), key=lambda p: p.name, reverse=True),
    ]

    script = next((p for p in candidates if p.exists()), None)
    if script is None:
        raise FileNotFoundError(
            f"Could not find RNQI stage {stage}. Checked: {candidates}"
        )

    module_name = f"_rnqi_stage_{stage.replace('-', '_')}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RNQI stage: {script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
