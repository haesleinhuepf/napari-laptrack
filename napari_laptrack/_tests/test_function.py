from typing import Callable

import napari

# from napari_laptrack import threshold, image_arithmetic
# add your tests here...

PLUGIN_NAME = "napari-laptrack"


def test_sample_data(make_napari_viewer: Callable[..., napari.Viewer]):
    viewer = make_napari_viewer()
    viewer.open_sample(PLUGIN_NAME, "Simple Tracks (2D)")
