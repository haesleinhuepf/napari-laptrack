import pandas as pd
from napari_plugin_engine import napari_hook_implementation
from napari_time_slicer import time_slicer
from napari_tools_menu import register_function


@napari_hook_implementation
def napari_experimental_provide_function():
    return [track_labels_centroid_based]


@register_function(menu="Tracking > Track labeled objects (centroid-based, LapTrack)")
def track_labels_centroid_based(
    image_layer: "napari.layers.Image",
    labels_layer: "napari.layers.Labels",
    viewer: "napari.Viewer",
):
    """
    Tracking particles in a 4D label image, based on the LapTrack library. Inspired by [1]

    See Also
    --------
    ..[1] https://github.com/yfukai/laptrack/blob/main/docs/examples/bright_spots.ipynb
    """
    from laptrack import LapTrack
    import napari_skimage_regionprops as nsr
    import napari_time_slicer as nts
    import numpy as np

    # Deal with input data of various formats and bring both in [T,Z,Y,X] shape/format, maybe with Z=1
    image = image_layer.data
    if len(image_layer.data.shape) == 2:
        print("Image data seems 2D/on-the-fly-processed, converting to 3D+t...")
        image_layer_4d = nts._function.convert_to_stack4d(image_layer, viewer)
        image = image_layer_4d.data
        image_name = image_layer_4d.name
        viewer.add_image(image, name=image_name)
    elif len(image_layer.data.shape) == 3:
        print("Image data seems 2D+t, converting to 3D+t...")
        image_layer_4d = nts._function.convert_to_2d_timelapse(image_layer, viewer)
        image = image_layer_4d.data
        image_name = image_layer_4d.name
        viewer.add_image(image, name=image_name)

    labels = labels_layer.data
    labels_layer_4d = labels_layer
    if len(labels_layer.data.shape) == 2:
        print("Labels data seems 2D/on-the-fly-processed, converting to 3D+t...")
        labels_layer_4d = nts._function.convert_to_stack4d(labels_layer, viewer)
        labels = labels_layer_4d.data
        labels_name = labels_layer_4d.name
        viewer.add_image(image, name=labels_name)
    elif len(labels_layer.data.shape) == 3:
        print("Labels data seems 2D+t, converting to 3D+t...")
        labels_layer_4d = nts._function.convert_to_2d_timelapse(labels_layer, viewer)
        labels = labels_layer_4d.data
        labels_name = labels_layer_4d.name
        viewer.add_image(image, name=labels_name)

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
            spots_df["centroid-2"] = 0
    else:
        spots_df = labels_layer_4d.features

    # LAP-based tracking
    lt = LapTrack(track_cost_cutoff=5**2, splitting_cost_cutoff=5**2)
    track_df, split_df, merge_df = lt.predict_dataframe(
        spots_df, ["centroid-0", "centroid-1", "centroid-2"], only_coordinate_cols=False
    )
    track_df = track_df.reset_index()

    # store results
    labels_layer_4d.features = track_df

    # show result as tracks
    split_merge_graph={}
    if not split_df.empty:
        split_merge_graph.update({
            row["child_track_id"]: [row["parent_track_id"]] for _, row in split_df.iterrows()
        })
    if not merge_df.empty:
        split_merge_graph.update({
            c_id: grp["parent_track_id"].to_list() for c_id, grp in merge_df.groupby("child_track_id")
        })
    viewer.add_tracks(
        track_df[["track_id", "frame", "centroid-2", "centroid-0", "centroid-1"]],
        graph=split_merge_graph,
        tail_length=50,
    )
    # show result as track-id-label image
    track_id_image = nsr.map_measurements_on_labels(labels_layer_4d, column="track_id")
    track_id_image = track_id_image.astype(np.uint32)
    viewer.add_labels(track_id_image)
    # show result as table
    nsr.add_table(labels_layer_4d, viewer)
