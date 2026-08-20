"""Canonical attribution geometry: AG native 2048, CBP native 2114.

What these pin is the thing a silent default would get wrong: a run that says nothing
must attribute AG's own 2048bp window and store all of it, while the adopted runs --
whose arrays on disk are 16384 -> crop 2114 -- must keep reporting 2114, because the
width every downstream stage is told has to be the width the array actually has.
"""
import pytest
from motifline.geometry import GeometryError, attr_width, geometry

CFG = {"geometry": {"cbp": {"in_window": 2114},
                    "ag": {"in_window": 2048, "crop": 2048}}}


def test_unspecified_run_gets_the_native_window_of_its_model():
    assert geometry(CFG, {"model": "ag"}) == {"in_window": 2048, "crop": 2048}
    assert geometry(CFG, {"model": "cbp"}) == {"in_window": 2114, "crop": 2114}
    assert attr_width(CFG, {"model": "ag"}) == 2048
    assert attr_width(CFG, {"model": "cbp"}) == 2114


def test_a_run_that_declares_the_old_geometry_keeps_it():
    # k562_ag_adopted and friends: the archive is a 16384bp attribution cropped to 2114,
    # and calling it 2048 would hand finemo a width its arrays do not have.
    r = {"model": "ag", "in_window": 16384, "crop": 2114}
    assert geometry(CFG, r) == {"in_window": 16384, "crop": 2114}
    assert attr_width(CFG, r) == 2114


def test_crop_wider_than_the_window_is_refused():
    with pytest.raises(GeometryError):
        geometry(CFG, {"model": "ag", "in_window": 2048, "crop": 2114})


def test_cbp_cannot_declare_a_crop():
    # ChromBPNet's input window IS its readout; a cropping cbp run is a different model.
    with pytest.raises(GeometryError):
        geometry(CFG, {"model": "cbp", "in_window": 2114, "crop": 2048})
    assert geometry(CFG, {"model": "cbp", "crop": 2114})["crop"] == 2114   # restating it is fine


def test_unknown_model_and_unset_canon_raise_rather_than_default():
    with pytest.raises(GeometryError):
        geometry(CFG, {"model": "enformer"})
    with pytest.raises(GeometryError):
        geometry({"geometry": {}}, {"model": "ag"})
