"""Sanity tests for :class:`runner.gran_runner_v2.GranRunnerV2`.

These tests do NOT actually train or exercise the model — they verify that
the YAML configs parse, contain the v2-specific fields, and that the runner
class is wired up correctly as a subclass of :class:`GranRunner`.

Instantiating ``GranRunnerV2`` requires a real dataset (via
:func:`utils.data_helper.create_graphs`) plus a writable ``save_dir`` for
the TensorBoard ``SummaryWriter``, so we only inspect the class definition
here instead of instantiating the runner.

Note: ``runner.gran_runner`` transitively imports :mod:`pyemd` through
``utils.dist_helper``. When pyemd is not installed (e.g. on dev boxes
without a C compiler) the runner import will fail with ModuleNotFoundError,
in which case the class-hierarchy test is skipped rather than errored.
"""
import importlib
import os

import pytest
import yaml
from easydict import EasyDict as edict


# Resolve config paths relative to the GRAN/ project root so the tests work
# regardless of where pytest is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))


def _load_config(rel_path):
    """Load a YAML config file relative to the GRAN/ project root.

    Args:
        rel_path: e.g. ``'config/gran_v2_grid.yaml'``.

    Returns:
        easydict.EasyDict: parsed config.
    """
    full_path = os.path.join(_PROJECT_ROOT, rel_path)
    with open(full_path, 'r') as f:
        return edict(yaml.safe_load(f))


def test_gran_v2_grid_config_parses():
    """config/gran_v2_grid.yaml should parse and have all required fields."""
    config = _load_config('config/gran_v2_grid.yaml')
    assert config.runner == 'GranRunnerV2'
    assert config.model.name == 'GRANv2'
    assert config.dataset.loader_name == 'GRANDataV2'
    assert config.model.num_attr_classes >= 1
    assert hasattr(config.model, 'use_gatv2')
    assert hasattr(config.model, 'gatv2_num_heads')
    assert hasattr(config.train, 'lambda_attr')
    # Smoke-check a couple of mirrored parent fields so we catch an
    # accidental YAML-structure divergence.
    assert config.dataset.name == 'grid'
    assert config.model.max_num_nodes == 361


def test_gran_v2_DB_config_parses():
    """config/gran_v2_DB.yaml should parse and have all required fields."""
    config = _load_config('config/gran_v2_DB.yaml')
    assert config.runner == 'GranRunnerV2'
    assert config.model.name == 'GRANv2'
    assert config.dataset.loader_name == 'GRANDataV2'
    assert config.model.num_attr_classes >= 1
    # FIRSTMM_DB uses richer node types than grid -> >=7 classes.
    assert config.model.num_attr_classes == 7
    assert config.dataset.name == 'FIRSTMM_DB'
    assert hasattr(config.train, 'lambda_attr')


def test_runner_v2_class_attributes():
    """GranRunnerV2 should inherit from GranRunner and override train/test.

    Instantiating the runner needs a full config + dataset on disk + a
    writable save_dir for TensorBoard, so this test only checks the class
    hierarchy / method existence.

    ``runner.gran_runner`` pulls in pyemd (via utils.dist_helper); if that
    optional C-backed dependency isn't installed in the current environment
    we skip rather than fail — the class-hierarchy guarantees still hold
    once the real training environment is set up.
    """
    try:
        gran_runner_mod = importlib.import_module('runner.gran_runner')
        gran_runner_v2_mod = importlib.import_module('runner.gran_runner_v2')
    except ModuleNotFoundError as exc:  # e.g. missing pyemd
        pytest.skip(
            "Skipping runner class-hierarchy check: optional runtime "
            "dependency unavailable in this environment ({}).".format(exc))

    GranRunner = gran_runner_mod.GranRunner
    GranRunnerV2 = gran_runner_v2_mod.GranRunnerV2

    assert issubclass(GranRunnerV2, GranRunner)
    assert hasattr(GranRunnerV2, 'train')
    assert hasattr(GranRunnerV2, 'test')
    # train/test should be overridden in the subclass (not merely inherited).
    assert GranRunnerV2.train is not GranRunner.train
    assert GranRunnerV2.test is not GranRunner.test
    # lambda_attr is a v2-specific instance attribute, set in __init__.
    # Verify the __init__ exists and references the new attribute.
    import inspect
    src = inspect.getsource(GranRunnerV2.__init__)
    assert 'lambda_attr' in src
