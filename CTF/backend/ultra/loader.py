import importlib.machinery
import importlib.util
from pathlib import Path

_NT2_CACHE = None


def _load_module_from_path(module_name, path):
    loader = importlib.machinery.SourceFileLoader(module_name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Missing loader for {path}")
    spec.loader.exec_module(module)
    return module


def load_nt2_module():
    global _NT2_CACHE
    if _NT2_CACHE is not None:
        return _NT2_CACHE

    backend_dir = Path(__file__).resolve().parents[1]
    path = backend_dir / "pick_flag_elite_ai-nt2.py"
    _NT2_CACHE = _load_module_from_path("pick_flag_elite_ai_nt2", path)
    return _NT2_CACHE
