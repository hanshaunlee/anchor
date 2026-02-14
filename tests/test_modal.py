"""Tests for Modal wrappers: modal_train, modal_train_elliptic, modal_app (structure and defaults)."""
import os

import pytest


def test_modal_train_module_structure() -> None:
    """ml.modal_train defines _REPO_ROOT and (if Modal installed) image and run_train."""
    from ml import modal_train
    assert hasattr(modal_train, "_REPO_ROOT")
    assert os.path.isabs(modal_train._REPO_ROOT)
    assert "anchor" in modal_train._REPO_ROOT.lower() or "ml" in modal_train._REPO_ROOT
    if modal_train.modal is not None:
        assert hasattr(modal_train, "_app")
        assert hasattr(modal_train, "run_train")
        assert hasattr(modal_train, "main")
        assert modal_train._app.name == "anchor-train"


def test_modal_train_elliptic_module_structure() -> None:
    """ml.modal_train_elliptic defines _REPO_ROOT and (if Modal installed) run_train_elliptic and main."""
    from ml import modal_train_elliptic
    assert hasattr(modal_train_elliptic, "_REPO_ROOT")
    assert os.path.isabs(modal_train_elliptic._REPO_ROOT)
    if modal_train_elliptic.modal is not None:
        assert hasattr(modal_train_elliptic, "_app")
        assert hasattr(modal_train_elliptic, "run_train_elliptic")
        assert hasattr(modal_train_elliptic, "main")
        assert modal_train_elliptic._app.name == "anchor-train-elliptic"


def test_modal_app_structure() -> None:
    """modal_app defines app and hello; app name is anchor."""
    import modal_app
    assert hasattr(modal_app, "app")
    assert hasattr(modal_app, "hello")
    assert hasattr(modal_app, "main")
    assert modal_app.app.name == "anchor"


def test_modal_app_hello_is_modal_function() -> None:
    """hello is a Modal function (has .remote for server execution)."""
    import modal_app
    # @app.function wraps hello; it's a Modal Function, not a plain callable.
    assert hasattr(modal_app.hello, "remote")
