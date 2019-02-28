#!/usr/bin/env python

from itertools import product
from copy import deepcopy
from hashlib import sha1

from scipy import ndimage
import numpy as np


def generate_noise_params(image, mask, erode=3, location=[], force=False,
                          mode="single"):
    """
    Creates paramaters for noise to be added within an image, conditioned by an
    image mask.

    Parameters
    ----------
    image : array_like
        Image matrix to be injected with noise
    mask : array_like
        Image matrix containing a mask for the region of interest when
        injecting noise. Non-zero elements are considered True. If there is no
        restriction on where noise can be placed, provide a mask with the same
        dimensions of the image above and intensity "1" everywhere.
    scale : boolean, optional
        Toggles multiplication scaling (True) or direct substitution (False) of
        the intensity parameter into the image, by default True.
    intensity : float, optional
        Value controlling the strength of noise injected into the image. If
        scale is True, this value will indicate a percentage change in existing
        image value. If False, then it will be substituted directly as the new
        voxel value. With scale by default True, this is by default 0.01,
        indicating a 1 percent increase in intensity.
    erode : int, optional
        Number of voxels to erode from the mask before choosing a location for
        noise injection. This is intended to be used with over-esimated masks.
        By default, 3 iterations of erosion will be performed.
    location : list or tuple of ints, optional
        Placement of the noise within the image. By default, none, meaning a
        random location will be generated within the mask.
    force : boolean, optional
        This forces the injection of noise when the location provided is
        outside of the mask. By default, False.
    mode : str, optional
        This determines where noise will be injected in the case of high
        dimensional images and lower-dimensional masks/locations. If "single",
        a single location in all the higher dimension will be selected,
        resulting in truly 1-voxel of noise. If "uniform", the voxel location
        determined in low dimensions will be given noise in all remaining
        dimensions, equivalent to an index of [i, j, k, :, ...], for instance.
        If "independent", noise will be added at a random location within the
        mask for each higher dimension; this option is mutually exclusive to
        the "location" parameter. By default, the "single" mode is used.

    Returns
    -------
    output : array_like
        The resultant image containing the data from "image" with the addition
        of 1-voxel noise.
    location : tuple of ints or list of tuples of ints
        Location(s) of injected noise within the image.
    """
    # Adding special case to erosion: when 0, don't erode.
    erode = int(erode)
    if erode:
        mask = ndimage.binary_erosion(mask, iterations=erode)
    mask_locs = np.where(mask > 0)
    mask_locs = list(zip(*mask_locs))

    # Verify valid mode
    modes = ["single", "uniform", "independent"]
    if mode not in modes:
        raise ValueError("Invalid mode. Options: single, uniform, independent")

    # Don't let user try to provide a location and use independent mode
    if mode == "independent" and location:
        raise ValueError("Cannot use 'location' and 'independent' mode.")

    # Verify mask is valid
    if len(mask.shape) > len(image.shape):
        raise ValueError("Mask can't have more dimensions than image.")
    if mask.shape != image.shape[0:len(mask.shape)]:
        raise ValueError("Mask must have same range for shared dimensions.")
    if not len(mask_locs):
        raise ValueError("Mask contains no locations - try eroding less.")

    # If a location is provided do some basic sanity checks
    if location:
        # Coerce location into tuple of integers
        location = tuple(int(l) for l in location)

        # Verify that the location is valid for the image
        if len(location) > len(image.shape):
            raise ValueError("Location can't have more dimensions than image.")

        if any(loc >= image.shape[idx]
               for idx, loc in enumerate(location)):
            raise ValueError("Location must be within the image extent.")

        if (location not in mask_locs) and not force:
            raise ValueError("Location must be within mask without --force.")

    # Generate location getter to either return the given loc or generate one
    def create_location_getter(location, mask_locs):
        def location_getter():
            if location:
                return location
            index = np.random.randint(0, high=len(mask_locs))
            return tuple(int(ml) for ml in mask_locs[index])
        return location_getter
    location_getter = create_location_getter(location, mask_locs)

    # Create noise sites for the image
    # If uniform, low dimensional location can be applied directly
    if mode == "uniform":
        loc = [location_getter()]

    # If single, generate a single index for remaining dimensions
    elif mode == "single":
        extra_loc = tuple(int(np.random.randint(0, high=n))
                          for n in image.shape[len(location_getter()):])
        loc = [location_getter() + extra_loc]

    # If independent, generate a location for each volume in all dimensions
    else:
        loc = []
        extra_range = image.shape[len(location_getter()):]
        extra_range = [list(int(rangeval) for rangeval in np.arange(er))
                       for er in extra_range]
        extra_locs = product(*extra_range)
        loc = [location_getter() + extra_loc for extra_loc in extra_locs]

    return loc


def apply_noise_params(image, locations, scale=True, intensity=0.01):
    # Create new container for image data
    image = deepcopy(image)

    # Generate noise injector to set or scale intensity of image
    def create_noise_injector(intensity, scale):
        def noise_injector(value):
            if scale:
                return value * (1 + intensity)
            return intensity
        return noise_injector
    noise_injector = create_noise_injector(intensity, scale)

    # For each location in the list of sets provided, add noise
    for loc in locations:
        image[loc] = noise_injector(image[loc])

    # Compute the hash of the new image
    image_hash = sha1(np.ascontiguousarray(image)).hexdigest()

    return (image, image_hash)
