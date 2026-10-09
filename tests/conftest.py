import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
STUB = Path(__file__).resolve().parent / "stub_model"


@pytest.fixture
def stub_path(tmp_path) -> Path:
    """A fresh copy of the stub model folder, so tests can write its zoo."""
    dst = tmp_path / "stub"
    shutil.copytree(STUB, dst, ignore=shutil.ignore_patterns("zoo", "__pycache__"))
    return dst


@pytest.fixture
def stub(stub_path):
    from core.model_folder import load_model_folder

    return load_model_folder(stub_path)
