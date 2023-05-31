from typing import Callable

import napari
import pytest
from napari_laptrack._function import _DATA

# from napari_laptrack import threshold, image_arithmetic
# add your tests here...

PLUGIN_NAME = "napari_laptrack"

def test_sample_data(make_napari_viewer: Callable[..., napari.Viewer]):
    viewer = make_napari_viewer(block_plugin_discovery=False)
    for k in _DATA.keys():
        viewer.open_sample(PLUGIN_NAME, k)
