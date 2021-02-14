#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 16:20:26 2019
Background subtraction

@author: Rafael Jaimes
raf@cardiacmap.com
v1: 2019-02-28
"""
import numpy as np
import cv2
from skimage import morphology


def split(img, horz):
    """Pulls in an image stack (T, X, Y). Splits in the horizontal direction while preserving Y and T
    Typically used to split dual wavelength images with a single sensor."""
    [num_frames, h, w] = img.shape
    left = img[:, :, 0:int(w * horz)]
    right = img[:, :, int(w * horz):]
    return left, right


def bg_remove(img, thresh, min_size, F0_locs):
    """Removes background noise using a threshold and binary mask

    Parameters
    ----------
    img : ndarray
        Image stack with shape (t, x, y)
    thresh : int
        Cutoff value, pixel values below this are removed
    min_size : int
        The smallest allowable connected component size
    F0_locs : ??
        ??

    Returns
    -------
    img_BG_removed : ndarray
        Image stack with binary mask applied
    mask : ndarray
        Binary mask applied to original image stack
    """

    # This function removes the background using thresholding
    # Adjust the input "thresh" value based on the histogram of the data.
    # A lower threshold will allow more pixels through.
    # First determine the standard deviation image
    F0_img = np.mean(img[F0_locs, :, :], axis=0, dtype='float32')
    ret1, imthres = cv2.threshold(F0_img, thresh, 1, cv2.THRESH_BINARY)  # only takes float32 and float64
    # Remove the small objects. set min_size of pixel group and connectivity (num of px away from heart)        
    imNoSmall = morphology.remove_small_objects(imthres.astype(bool), min_size, connectivity=1)  # min_size=512 works
    # use MORPH_CLOSE to fill the gaps
    kernel = np.ones((15, 15), dtype="uint8")  # increase the kernel size to fill bigger holes
    imFilled = cv2.morphologyEx(imNoSmall.astype("uint8"), cv2.MORPH_CLOSE, kernel)
    mask = imFilled.astype(bool)
    img_BG_removed = np.zeros(np.shape(img), dtype="uint16")
    for p in range(0, np.size(img, axis=0)):
        img_BG_removed[p, :, :] = img[p, :, :] * mask
    return img_BG_removed, mask


def boxblur(img, kernel_size):
    """Applies a convolution filter/box blur to an image stack

    Parameters
    ----------
    img : ndarray
        Image stack with shape (t, x, y)
    kernel_size : int
        Pixel size of square kernel to use in convolution

    Returns
    -------
    img_blur : ndarray
        Image stack with box blur applied
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ^ 2)  # Smoothing Kernel
    img_blur = np.zeros(np.shape(img), dtype="float32")
    for p in range(0, np.size(img, axis=0)):
        img_blur[p, :, :] = cv2.filter2D(img[p, :, :], -1, kernel)  # convolve the image
    return img_blur


def rotate(img):
    num_frames, height, width = img[:, :, :].shape  # Get the height and width from first frame
    rot = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
    img_rot = np.zeros([num_frames, width, height], dtype="uint16")
    for p in range(0, np.size(img, axis=0)):
        img_rot[p, :, :] = cv2.warpAffine(img[p, :, :], rot, (width, height))
    return img_rot
