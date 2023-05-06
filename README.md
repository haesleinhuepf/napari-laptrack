# napari-laptrack

[![License](https://img.shields.io/pypi/l/napari-laptrack.svg?color=green)](https://github.com/haesleinhuepf/napari-laptrack/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-laptrack.svg?color=green)](https://pypi.org/project/napari-laptrack)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-laptrack.svg?color=green)](https://python.org)
[![tests](https://github.com/haesleinhuepf/napari-laptrack/workflows/tests/badge.svg)](https://github.com/haesleinhuepf/napari-laptrack/actions)
[![codecov](https://codecov.io/gh/haesleinhuepf/napari-laptrack/branch/master/graph/badge.svg)](https://codecov.io/gh/haesleinhuepf/napari-laptrack)
[![Development Status](https://img.shields.io/pypi/status/napari-laptrack.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-laptrack)](https://napari-hub.org/plugins/napari-laptrack)

Tracking particles in Napari, using the [LapTrack](https://laptrack.readthedocs.io/en/latest/usage.html) library.
This plugin is young and has just limited functionality. Contributions are welcome.

## Installation instructions

It is recommended to use this napari-plugin together with [devbio-napari](https://github.com/haesleinhuepf/devbio-napari).
Install both using mamba-forge ([download here](https://github.com/conda-forge/miniforge#mambaforge)) by running these commands line-by line from the terminal:

```
mamba create --name napari-laptrack-env -c conda-forge python=3.9 devbio-napari
```

```
mamba activate napari-laptrack-env
```

```
pip install napari-laptrack
```

## Usage

The starting point for napari-laptrack is a 4D image layer and a corresponding labels layer.
The following procedure demonstrates how to start from a 2D+t image stack, convert it in the right format and segment the labels.
Afterwards, napari-laptrack is demonstrated. Depending on your input data, you may skip some of the initial steps.

### Example data

We demonstrate the procedure using the example dataset `File > Open Samples > clesperanto > CalibZAPWfixed` which should be available if you installed [devbio-napari](https://github.com/haesleinhuepf/devbio-napari).
You can also download it from [zenodo](https://zenodo.org/record/5090508#.ZDQZ9nZBxaQ).

### 4D+t input data.

In case your image data comes as 3D-stack, you must convert it in the format 4D+t with shape [t,1,y,x] first.
You can do this using the menu `Tools > Utilities > Convert 3D stack to 2d+t timelapse`, which is part of the [napari-time-slicer](https://www.napari-hub.org/plugins/napari-time-slicer) plugin.
It will create a new layer named `2D+t <original name>`. After this conversion, you can delete the original image layer, which is recommended to avoid confusion due to too many layers.
For deleting the original layer, select it and hit the trash-bin button.

![img.png](https://github.com/haesleinhuepf/napari-laptrack/raw/main/docs/convert2d_t.png)

### Object segmentation

Various segmentation algorithms are available in Napari (see the [Napari-hub](https://www.napari-hub.org/?search=segmentation&sort=relevance&page=1)).
In principle all algorithms are compatible if they produce a 3D+t label image as result.
In this tutorial, we use the [Voronoi-Otsu-Labeling algorithm](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/20_image_segmentation/11_voronoi_otsu_labeling.html) implemented using [clesperanto](https://github.com/clEsperanto/pyclesperanto_prototype).
It is available from the menu `Tools > Segmentation / labeling`.

![img.png](https://github.com/haesleinhuepf/napari-laptrack/raw/main/docs/labeling_vol.png)

### Tracking labeled objects

Now that we have a 3D+t image and a corresponding label-image, we can start tracking the objects.
Centroid-based tracking is available from the menu `Tracking > Track labeled objects (centroid-based, LapTrack)`.
After tracking, multiple new layers will be added to Napari, which are explained in detail below.
Furthermore, a table will open containing the columns `centroid-0/1/2` with spatial positions of the labels.
The table also contain colums `label`, `frame` and `track_id`.
All labels which belong to the same track, but to different frames, have the same `track_id`.
In some cases, also layers named `Stack 4D <original layer name>` are created. This is done to store the labels which were analysed. These layers are technically duplicates of the original layers which were computed on-the-fly.

![img.png](https://github.com/haesleinhuepf/napari-laptrack/raw/main/docs/result.png)

### The Tracks layer

The tracks layer visualizes the travel path of the labels' centroids over time. [Read more about the Tracks layer in the Napari documentation](https://napari.org/stable/howtos/layers/tracks.html).

![img.png](https://github.com/haesleinhuepf/napari-laptrack/raw/main/docs/tracks_layer.png)

### The Track-ID image

One result of the plugin is a Track-ID image. This is a label image where objects have the same label / color over time.
This image is not suited for many quantitative label-measurment methods because it is non-sequentially labeled.

As example, two subsequent frames are shown:

![img.png](https://github.com/haesleinhuepf/napari-laptrack/raw/main/docs/track_id_image_0.png)

![img.png](https://github.com/haesleinhuepf/napari-laptrack/raw/main/docs/track_id_image_1.png)

## Similar and related plugins

There are other napari-plugins and python packages which allow tracking particles, visualizing tracking data and quantiative measurements of tracks:

- [arboretum](https://github.com/lowe-lab-ucl/arboretum)
- [btrack](https://github.com/quantumjot/btrack)
- [ultrack](https://github.com/royerlab/ultrack)
- [napari-stracking](https://www.napari-hub.org/plugins/napari-stracking)
- [napari-tracks-reader](https://www.napari-hub.org/plugins/napari-tracks-reader)
- [vollseg-napari-trackmate](https://www.napari-hub.org/plugins/vollseg-napari-trackmate)
- [palmari](https://www.napari-hub.org/plugins/palmari)
- [napari-amdtrk](https://www.napari-hub.org/plugins/napari-amdtrk)

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-laptrack" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

<!-- prettier-ignore-start -->
[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/haesleinhuepf/cookiecutter-napari-assistant-plugin
[file an issue]: https://github.com/haesleinhuepf/napari-laptrack/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
<!-- prettier-ignore-end -->
