"""Tests-level conftest.

Stubs `src.models` before any test imports `src.zoom`, so the pure geometric
tests don't require torch / transformers / a GPU.
"""

import sys
import types

if "src.models" not in sys.modules:
    stub = types.ModuleType("src.models")
    stub.run_vlm = lambda *args, **kwargs: None
    sys.modules["src.models"] = stub
