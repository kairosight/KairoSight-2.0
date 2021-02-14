#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:08:46 2019
Opens a TIFF Stack of the form (t, x, y).
User should provide the path and file name (fname)
Outputs the image stack and calculated period between frames (dt)
Tested with MetaMorph 7.5 TIFF Specification

Note: Returns dt in milliseconds.

@author: Rafael Jaimes
raf@cardiacmap.com
v1: 2019-02-26 
"""

from imageio import volread
import exifread
import pandas as pd


def tifopen(path, fname):
    file = open(path + fname, 'rb')
    tags = exifread.process_file(file)  # Read EXIF data
    img = volread(path + fname)  # Read image data, keep this second because it closes the file after reading

    try:  # Check if it's a MetaMorph 7.5 TIFF
        start_time = pd.to_datetime(tags['Image DateTime'].values)
        [num_frames, _, _] = img.shape
        end_time = pd.to_datetime(tags['IFD ' + str(num_frames - 1) + ' DateTime'].values)
        duration = end_time - start_time
        duration_ms = float(duration.microseconds / 1000) + float(duration.seconds * 1000)  # Days attribute is ignored
        dt = float(duration_ms / (num_frames - 1))  # Period
    except:
        try:  # Check if it's an OME TIFF
            image_description = tags['Image ImageDescription'].printable
            dt_loc = image_description.find('Plane DeltaT=')
            dt = float(image_description[dt_loc + 14:dt_loc + 24]) / 1000  # 10 significant figures for dt
        except:
            dt = float('nan')
            print('Could not determine frame rate from timestamps.')

    return img, dt
