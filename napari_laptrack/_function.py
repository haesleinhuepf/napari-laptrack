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


def _simple_tracks():
    from laptrack.datasets import simple_tracks
    import numpy as np
    data = simple_tracks()[["frame","position_y","position_x"]].values

    yrange = (150,250)
    xrange = (150,250)

    track_image = np.zeros((
        int(data[:,0].max())+1,
        yrange[1] - yrange[0],
        xrange[1] - xrange[0],
    ))
    yy, xx = np.meshgrid(
        np.arange(track_image.shape[1]),
        np.arange(track_image.shape[2]),
        indexing='ij'
    )
    data[:,1] = data[:,1] - yrange[0]
    data[:,2] = data[:,2] - xrange[0]
    for d in data:
        track_image[int(d[0])] = np.exp(-((yy-d[1])**2 + (xx-d[2])**2))
    return [
        (track_image, {"name": "Simple Tracks (2D) Images"}, "image"),
        (data, {"name": "Simple Tracks (2D)","face_color":"red"}, "points"),
    ]

def _cell_segmentation():
    from laptrack.datasets import cell_segmentation
    data, label = cell_segmentation()
    return [
        (data, {"name": "C2C12 Cells (2D)"}, "image"),
        (label, {"name": "C2C12 Cells (2D) Labels"}, "labels"),
    ]

@napari_hook_implementation
def napari_provide_sample_data():
    from laptrack.datasets import bright_brownian_particles, \
        mouse_epidermis
    
    return {
        "Simple Tracks (2D)": _simple_tracks,
        "Bright Brownian Particles (2D)": \
            lambda : [(bright_brownian_particles(), 
                       {"name": "Bright Brownian Particles (2D)"})],
        "C2C12 Cells (2D)": _cell_segmentation,
        "Mouse Epidermis (2D)": \
            lambda : [(mouse_epidermis(), 
                       {"name": "Mouse Epidermis (2D)"},
                       "labels")],

    }
