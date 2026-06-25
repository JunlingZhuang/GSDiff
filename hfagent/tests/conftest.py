import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root -> import hfagent.*

_OUT_ROOT = Path(__file__).resolve().parent.parent / "out" / "tests"


@pytest.fixture
def out_dir(request):
    """Persistent output directory under hfagent/out/tests/<test_name>/ for inspecting artifacts."""
    path = _OUT_ROOT / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    return path
