#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import os

from setuptools import find_packages
from setuptools import setup


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()


# Add your dependencies in requirements.txt
# Note: you can add test-specific requirements in tox.ini
requirements = []
with open("requirements.txt") as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

setup(
    name="napari-laptrack",
    author="Robert Haase",
    author_email="robert.haase@tu-dresden.de",
    license="BSD-3",
    url="https://github.com/haesleinhuepf/napari_laptrack",
    description="Tracking particles in Napari, based on the LapTrack library",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=requirements,
    version="0.1.0",
    setup_requires=["setuptools_scm"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Framework :: napari",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ],
    entry_points={
        "napari.plugin": [
            "napari_laptrack = napari_laptrack",
        ],
    },
)
