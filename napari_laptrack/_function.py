import pandas as pd
from napari_plugin_engine import napari_hook_implementation
from napari_time_slicer import time_slicer
from napari_tools_menu import register_function


@napari_hook_implementation
def napari_experimental_provide_function():  # noqa: D103
    return [track_labels_centroid_based]

def _check_and_convert_layers_to_4d(layer,viewer):
    import napari_time_slicer as nts
    # Deal with input data of various formats and bring both in [T,Z,Y,X] shape/format, maybe with Z=1
    if len(layer.data.shape) == 3:
        return nts._function.convert_to_2d_timelapse(layer, viewer)
    elif len(layer.data.shape) == 4:
        return layer
    else:
        raise ValueError(f"image shape {layer.data.shape} not supported")

@register_function(menu="Tracking > Track labeled objects (centroid-based, LapTrack)")
def track_labels_centroid_based(
    image_layer: "napari.layers.Image",
    labels_layer: "napari.layers.Labels",
    viewer: "napari.Viewer",
):
    """
    Tracking particles in a 4D label image, based on the LapTrack library. Inspired by [1].

    See Also
    --------
    ..[1] https://github.com/yfukai/laptrack/blob/main/docs/examples/bright_spots.ipynb
    """
    from laptrack import LapTrack
    from laptrack.data_conversion import convert_split_merge_df_to_napari_graph
    import napari_skimage_regionprops as nsr
    import numpy as np

    image_layer_4d = _check_and_convert_layers_to_4d(image_layer, viewer)
    viewer.add_layer(image_layer_4d)
    image = image_layer_4d.data

    labels_layer_4d = _check_and_convert_layers_to_4d(labels_layer, viewer)
    viewer.add_layer(labels_layer_4d)
    labels = labels_layer_4d.data

    # determine centroids
    if (
        labels_layer_4d.features is None
        or "centroid-0" not in labels_layer_4d.features.keys()
    ):
        print(
            "No centroids found in measured features; determining centroid using napari-skimage-regionprops..."
        )
        measurements = nsr.regionprops_table_all_frames(image, labels, position=True)
        spots_df = pd.DataFrame(measurements)
        if "centroid-2" not in spots_df.keys():
            spots_df.rename(columns={"centroid-0": "centroid-1", "centroid-1": "centroid-2"}, inplace=True)
            spots_df["centroid-0"] = 0
    else:
        spots_df = labels_layer_4d.features

    # LAP-based tracking
    lt = LapTrack(track_cost_cutoff=5**2, splitting_cost_cutoff=5**2)
    track_df, split_df, merge_df = lt.predict_dataframe(
        spots_df, ["centroid-0", "centroid-1", "centroid-2"], only_coordinate_cols=False
    )
    track_df = track_df.reset_index()
    track_df["track_id"] = track_df["track_id"]+1
    track_df["tree_id"] = track_df["tree_id"]+1
    if not split_df.empty:
        split_df["parent_track_id"] = split_df["parent_track_id"]+1
        split_df["child_track_id"] = split_df["child_track_id"]+1
    if not merge_df.empty:
        merge_df["parent_track_id"] = merge_df["parent_track_id"]+1
        merge_df["child_track_id"] = merge_df["child_track_id"]+1

    # store results
    labels_layer_4d.features = track_df
    # show result as tracks
    split_merge_graph=convert_split_merge_df_to_napari_graph(split_df,merge_df)
    viewer.add_tracks(
        track_df[["track_id", "frame", "centroid-0", "centroid-1", "centroid-2"]],
        graph=split_merge_graph,
        tail_length=50,
    )

    # show result as track-id-label image
    track_id_image = nsr.map_measurements_on_labels(labels_layer_4d, column="track_id")
    track_id_image = track_id_image.astype(np.uint32)
    viewer.add_labels(track_id_image)
    # show result as table
    nsr.add_table(labels_layer_4d, viewer)


def _simple_tracks():
    from laptrack.datasets import simple_tracks
    import numpy as np

    data = simple_tracks()[["frame", "position_y", "position_x"]].values

    yrange = (150, 250)
    xrange = (150, 250)

    track_image = np.zeros(
        (
            int(data[:, 0].max()) + 1,
            yrange[1] - yrange[0],
            xrange[1] - xrange[0],
        )
    )
    yy, xx = np.meshgrid(
        np.arange(track_image.shape[1]), np.arange(track_image.shape[2]), indexing="ij"
    )
    data[:, 1] = data[:, 1] - yrange[0]
    data[:, 2] = data[:, 2] - xrange[0]
    for d in data:
        track_image[int(d[0])] = np.exp(-((yy - d[1]) ** 2 + (xx - d[2]) ** 2))
    return [
        (track_image, {"name": "Simple Tracks (2D) Images"}, "image"),
        (data, {"name": "Simple Tracks (2D)", "face_color": "red"}, "points"),
    ]


def _cell_segmentation():
    from laptrack.datasets import cell_segmentation

    data, label = cell_segmentation()
    return [
        (data, {"name": "Cell Segmentation (2D)"}, "image"),
        (label, {"name": "Cell Segmentation (2D) Labels"}, "labels"),
    ]


def _bright_brownian_particles():
    from laptrack.datasets import bright_brownian_particles

    return [(bright_brownian_particles(), {"name": "Bright Brownian Particles (2D)"})]


def _mouse_epidermis():
    from laptrack.datasets import mouse_epidermis

    return [(mouse_epidermis(), {"name": "Mouse Epidermis (2D)"}, "labels")]

def _HL60_3D_synthesized():
    from laptrack.datasets import HL60_3D_synthesized

    data, label = HL60_3D_synthesized()
    return [
        (data, {"name": "HL60 Synthesized (3D)"}, "image"),
        (label, {"name": "HL60 Synthesized (3D) Labels"}, "labels"),
    ]

_DATA = {
    "simple_tracks": {"data": _simple_tracks, "display_name": "Simple Tracks (2D)"},
    "bright_brownian_particles": {
        "data": _bright_brownian_particles,
        "display_name": "Bright Brownian Particles (2D)",
    },
    "cell_segmentation": {
        "data": _cell_segmentation,
        "display_name": "Cell Segmentation (2D)",
    },
    "mouse_epidermis": {
        "data": _mouse_epidermis,
        "display_name": "Mouse Epidermis (2D)",
    },
    "HL60_3D_synthesized": {
        "data": _HL60_3D_synthesized,
        "display_name": "HL60 Synthesized (3D)",
    },
}

# May be useful in npe2 migration later
# globals().update({k: v['data'] for k, v in _DATA.items()})

@napari_hook_implementation
def napari_provide_sample_data():
    return _DATA
