"""Test that DetaNet can be imported via the ar_mto bridge."""

import pytest


def test_import_ar_mto():
    import ar_mto
    assert ar_mto.__version__ == "0.2.0"


def test_locate_detanet():
    from ar_mto.detanet_bridge import _locate_detanet, get_detanet_path

    path = _locate_detanet()
    assert path.exists()
    assert (path / "detanet_model" / "__init__.py").exists()
    assert get_detanet_path() == path


def test_import_detanet():
    from ar_mto.detanet_bridge import import_detanet

    DetaNet = import_detanet()
    assert DetaNet.__name__ == "DetaNet"


def test_make_latent_detanet():
    from ar_mto.detanet_bridge import make_latent_detanet

    model = make_latent_detanet(num_block=1)
    assert hasattr(model, "forward")
    assert model.out_type == "latent"
    assert model.summation is False
