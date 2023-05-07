from typing import Callable

import napari
import pytest


# from napari_laptrack import threshold, image_arithmetic
# add your tests here...

PLUGIN_NAME = "napari_laptrack"

@pytest.mark.skip(reason="https://github.com/napari/napari/issues/5810")
def test_sample_data(make_napari_viewer: Callable[..., napari.Viewer]):
    viewer = make_napari_viewer()
    viewer.open_sample(PLUGIN_NAME, "simple_tracks")
