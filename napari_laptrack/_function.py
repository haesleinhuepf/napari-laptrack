import pandas as pd
import numpy as np
from napari_plugin_engine import napari_hook_implementation
from napari_time_slicer import time_slicer
from napari_tools_menu import register_function
from magicgui import magicgui
from typing import Literal, Annotated
from sklearn.metrics import confusion_matrix
from itertools import product


@napari_hook_implementation
def napari_experimental_provide_function():
    return [track_labels_centroid_based, track_labels_overlap_based]


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
    else:
        labels_name = labels_layer.name
        

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
    lt = LapTrack(track_cost_cutoff=5**2)
    track_df, _, _ = lt.predict_dataframe(
        spots_df, ["centroid-0", "centroid-1", "centroid-2"], only_coordinate_cols=False
    )
    track_df = track_df.reset_index()

    # store results
    labels_layer_4d.features = track_df

    # show result as tracks
    viewer.add_tracks(
        track_df[["track_id", "frame", "centroid-2", "centroid-0", "centroid-1"]],
        tail_length=50,
    )
    # show result as track-id-label image
    track_id_image = nsr.map_measurements_on_labels(labels_layer_4d, column="track_id")
    track_id_image = track_id_image.astype(np.uint32)
    viewer.add_labels(track_id_image)
    # show result as table
    nsr.add_table(labels_layer_4d, viewer)

@magicgui
@register_function(menu="Tracking > Track labeled objects (overlap-based, LapTrack)")
def track_labels_overlap_based(
    image_layer: "napari.layers.Image",
    labels_layer: "napari.layers.Labels",
    viewer: "napari.Viewer",
    metric: Literal["overlap_metric"] = "overlap_metric",
    track_cost_cutoff: Annotated[float, {"min": 0, "max": 500}] = 0.9,
    gap_closing_max_frame_count: Annotated[int, {"min": 0, "max": 10}] = 1,
    splitting_cost_cutoff: Annotated[float, {"min": 0, "max": 500}] = 0.9,
):
    """
    Tracking particles in a 4D label image, based on the LapTrack library. Inspired by [1]

    See Also
    --------
    ..[1] https://github.com/yfukai/laptrack/blob/main/docs/examples/overlap_tracking.ipynb
    """
    import napari_time_slicer as nts
    from laptrack import LapTrack
    from laptrack.metric_utils import LabelOverlap
    from functools import partial
    import napari_skimage_regionprops as nsr

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
    else:
        labels_name = labels_layer.name

    # Compute overlap between labels in consecutive frames and store results in a dataframe
    overlap_records = []
    for f in np.arange(labels.shape[0] - 1):

        # Get unique labels in consecutive frames
        unique_labels1 = np.unique(labels[f])
        unique_labels2 = np.unique(labels[f + 1])
        print(f'Computing overlap of frame {f} and {f + 1}')
        overlap_records += calculate_overlap(labels[f].ravel(), labels[f + 1].ravel(), unique_labels1, unique_labels2, f)
    
    # Convert list of dictionaries to dataframe
    overlap_df = pd.DataFrame.from_records(overlap_records)
    overlap_df = overlap_df.set_index(['frame', 'label1', 'label2']).copy()

    # Load/compute centroid positions of labels in each frame and store results in a dataframe
    if (
        labels_layer_4d.features is None
        or "centroid-0" not in labels_layer_4d.features.keys()
    ):
        print(
            "No centroids found in measured features; determining centroid using napari-skimage-regionprops..."
        )
        measurements = nsr.regionprops_table_all_frames(image, labels, position=True)
        coordinate_df = pd.DataFrame(measurements)
        if "centroid-2" not in coordinate_df.keys():
            coordinate_df["centroid-2"] = 0
    else:
        coordinate_df = labels_layer_4d.features

    # Instantiate LapTrack object
    if metric == "overlap_metric":
        overlap_metric_partial = partial(overlap_metric, overlap_df=overlap_df)
        lt = LapTrack(
            track_dist_metric=overlap_metric_partial,
            track_cost_cutoff=track_cost_cutoff,
            gap_closing_dist_metric=overlap_metric_partial,
            gap_closing_max_frame_count=gap_closing_max_frame_count,
            splitting_dist_metric=overlap_metric_partial,
            splitting_cost_cutoff=splitting_cost_cutoff,
        )
    
    # Compute tracks
    track_df, split_df, _ = lt.predict_dataframe(
        coordinate_df, coordinate_cols=["frame", "label"], only_coordinate_cols=False
    )
    track_df = track_df.reset_index()

    # Re-label objects in the images based on tracks and lineage trees
    new_labels = np.zeros_like(labels)
    for tree_id, grp in track_df.groupby("tree_id"):
        for _, row in grp.iterrows():
            frame = int(row["frame"])
            label_obj = int(row["label"])
            new_labels[frame][labels[frame] == label_obj] = tree_id + 1
    # Add new labels to viewer        
    viewer.add_labels(new_labels, name=labels_name + "_tracked")

    # Show tracks in napari
    viewer.add_tracks(
    track_df[["track_id", "frame", "centroid-2", "centroid-0", "centroid-1"]].values,
    graph={row["child_track_id"]: row["parent_track_id"] for _, row in split_df.iterrows()},
    tail_length=1)

    # show result as track-id-label image
    #track_id_image = nsr.map_measurements_on_labels(labels_layer_4d, column="track_id")
    #track_id_image = track_id_image.astype(np.uint32)
    #viewer.add_labels(track_id_image)
    # show result as table
    #nsr.add_table(labels_layer_4d, viewer)


def calculate_overlap(label1:np.array, label2:np.array, unique_labels1:np.array, unique_labels2:np.array, frame:int) ->list:
    """ Compute overlap, IoU and specific ratios between labels in two consecutive timeframes.

        Parameters:
        label1 : np.array
            Labels in timeframe f in the form of a 1D array.
        label2 : np.array
            Labels in timeframe f+1 in the form of a 1D array.
        unique_labels1 : np.array
            Unique labels in timeframe f in the form of a 1D array.
        unique_labels2 : np.array
            Unique labels in timeframe f+1 in the form of a 1D array.
        frame : int
            Specific index in the timelapse of images.
    """
    overlap_records = []
    keys = ("frame", "label1", "label2", "overlap", "iou", "ratio_1", "ratio_2")

    # Compute matrix consisting of the overlap between labels in two consecutive timeframes
    overlap_matrix = confusion_matrix(label1, label2)
    unique_labels_combined = np.union1d(unique_labels1, unique_labels2)

    # Compute indices of unique labels of specific timeframe in the combined array
    _, _, unique_labels1_indices = np.intersect1d(unique_labels1, unique_labels_combined, return_indices=True)
    _, _, unique_labels2_indices = np.intersect1d(unique_labels2, unique_labels_combined, return_indices=True)

    for index_1, index_2 in product(unique_labels1_indices, unique_labels2_indices):
        # Skip background label
        if index_1 == 0 or index_2 == 0:
            continue
        # Skip labels that do not overlap
        if overlap_matrix[index_1, index_2] == 0:
            continue
        overlap = overlap_matrix[index_1, index_2]
        b1_sum = np.sum(overlap_matrix[index_1, :])
        b2_sum = np.sum(overlap_matrix[:, index_2])
        union = b1_sum + b2_sum - overlap
        overlap_records.append(tuple([frame, unique_labels_combined[index_1], unique_labels_combined[index_2], overlap, overlap / union, overlap / b1_sum, overlap / b2_sum]))
    overlap_records = [dict(zip(keys, values)) for values in overlap_records]
    return overlap_records


def overlap_metric(c1, c2, overlap_df):
    (frame1, label1), (frame2, label2) = c1, c2
    if frame1 == frame2 + 1:
        tmp = (frame1, label1)
        (frame1, label1) = (frame2, label2)
        (frame2, label2) = tmp
    assert frame1 + 1 == frame2
    ind = (frame1, label1, label2)
    if ind in overlap_df.index:
        ratio_2 = overlap_df.loc[ind]["ratio_2"]
        return 1 - ratio_2
    else:
        return 1