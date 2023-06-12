from typing import Callable

import pytest

import napari
import pandas as pd
import numpy as np
import napari_skimage_regionprops as nsr
from laptrack import LapTrack
from laptrack.data_conversion import convert_split_merge_df_to_napari_graph

from napari_laptrack._function import _DATA
from napari_laptrack._function import track_labels_centroid_based

# from napari_laptrack import threshold, image_arithmetic
# add your tests here...

PLUGIN_NAME = "napari_laptrack"

def test_sample_data(make_napari_viewer: Callable[..., napari.Viewer]):
    viewer = make_napari_viewer(block_plugin_discovery=False)
    for k in _DATA.keys():
        viewer.open_sample(PLUGIN_NAME, k)

@pytest.mark.parametrize("dimension,data", [(2, "cell_segmentation"), (3, "HL60_3D_synthesized")])
def test_tracking(make_napari_viewer: Callable[..., napari.Viewer], dimension, data):
    viewer = make_napari_viewer(block_plugin_discovery=False)
    image_layer, labels_layer = viewer.open_sample(PLUGIN_NAME, data)
    track_labels_centroid_based(image_layer,labels_layer,viewer) 

    # layer existence
    assert labels_layer.visible == False

    if dimension == 2:
        assert image_layer.visible == False
        assert "2d+t " + image_layer.name in viewer.layers
        assert "2d+t " + labels_layer.name in viewer.layers
        assert viewer.layers["2d+t " + labels_layer.name].visible == False
        image = viewer.layers["2d+t "+image_layer.name].data
        labels = viewer.layers["2d+t "+labels_layer.name].data
    else:
        image = image_layer.data
        labels = labels_layer.data

    assert "LapTrack (centroid-based) " + image_layer.name in viewer.layers
    assert "LapTrack (centroid-based) labels " + labels_layer.name in viewer.layers
    track_layer = viewer.layers["LapTrack (centroid-based) " + image_layer.name]
    labels_track_layer = viewer.layers["LapTrack (centroid-based) labels " + labels_layer.name]

    measurements = nsr.regionprops_table_all_frames(image, labels, position=True)
    spots_df = pd.DataFrame(measurements)
    if dimension == 2:
        spots_df.rename(columns={"centroid-0": "centroid-1", "centroid-1": "centroid-2"}, inplace=True)
        spots_df["centroid-0"] = 0
    lt = LapTrack(track_cost_cutoff=5**2, splitting_cost_cutoff=5**2)
    track_df, split_df, merge_df = lt.predict_dataframe(
        spots_df, ["centroid-0", "centroid-1", "centroid-2"], only_coordinate_cols=False, index_offset=1
    )
    data = track_df.reset_index()[["track_id", "frame", "centroid-0", "centroid-1", "centroid-2"]].values
    split_merge_graph=convert_split_merge_df_to_napari_graph(split_df,merge_df)

    assert len(track_layer.data) == len(data)
    for d in data:
        assert np.any(np.all(track_layer.data == d,axis=1))
    assert track_layer.graph == split_merge_graph



