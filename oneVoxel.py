#!/usr/bin/env python

from argparse import ArgumentParser
from itertools import product
from copy import deepcopy
import os.path as op
import operator
import json
import uuid

from nilearn import image as nilimage
from scipy import ndimage
import nibabel as nib
import numpy as np


def one_voxel_noise(image, mask, scale=True, intensity=0.01, erode=3,
                    location=[], force=False, mode="single"):
    """
    Adds noise to a single voxel within an image, conditioned by an image mask.

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
    # Create new container for image data
    image = deepcopy(image)

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

    # Generate noise injector to set or scale intensity of image
    def create_noise_injector(intensity, scale):
        def noise_injector(value):
            if scale:
                return value * (1 + intensity)
            return intensity
        return noise_injector
    noise_injector = create_noise_injector(intensity, scale)

    # Apply noise to image
    # If uniform, low dimensional location can be applied directly
    if mode == "uniform":
        loc = location_getter()
        image[loc] = noise_injector(image[loc])

    # If single, generate a single index for remaining dimensions
    elif mode == "single":
        extra_loc = tuple(int(np.random.randint(0, high=n))
                          for n in image.shape[len(location_getter()):])
        loc = location_getter() + extra_loc
        image[loc] = noise_injector(image[loc])

    # If independent, generate a location for each volume in all dimensions
    else:
        # In this case only, the location will be a list of positions
        loc = []
        extra_range = image.shape[len(location_getter()):]
        extra_range = [list(int(rangeval) for rangeval in np.arange(er))
                       for er in extra_range]
        extra_locs = product(*extra_range)
        for extra_loc in extra_locs:
            loc += [location_getter() + extra_loc]
            image[loc[-1]] = noise_injector(image[loc[-1]])

    return (image, loc)


def make_descriptor(parser, arguments=None):
    import boutiques.creator as bc

    desc = bc.CreateDescriptor(parser, execname=op.basename(__file__),
                               tags={"domain": ["neuroinformatics",
                                                "image processing",
                                                "mri", "noise"]})
    basename = op.splitext(__file__)[0]
    desc.save(basename + ".json")

    if arguments is not None:
        invo = desc.createInvocation(arguments)
        invo.pop("boutiques")

        with open(basename + "_inputs.json", "w") as fhandle:
            fhandle.write(json.dumps(invo, indent=4))


def main():
    parser = ArgumentParser(__file__,
                            description="Adds noise to a single voxel within an"
                                        " image, conditioned by an image mask.")
    parser.add_argument("image_file",
                        help="Nifti image to be injected with one-voxel noise. "
                             "Default behaviour is that this will be done at a "
                             "random location within an image mask.")
    parser.add_argument("mask_file",
                        help="Nifti image containing a binary mask for the "
                             "input image. The noise location will be selected "
                             "randomly within this mask, unless a location is "
                             "provided.")
    parser.add_argument("output_directory",
                        help="Path for where the resulting Nifti image with one"
                             " voxel noise will be stored.")
    parser.add_argument("--scale", "-s", action="store_true",
                        help="Dictates the way in which noise is aplpied to the"
                             " image. If not set, the value specified with the "
                             "intensity flag will be set to the new value. If "
                             "set, the intensity value will be multiplied by "
                             "the original image value at the target location.")
    parser.add_argument("--intensity", "-i", action="store", type=float,
                        default=0.01,
                        help="The intensity of the noise to be injected in the "
                             "image. Default value is 0.01 so specifying the "
                             "scale flag alone will result in a 1%% intensity "
                             "change at the target location.")
    parser.add_argument("--erode", "-e", action="store", type=int,
                        default=3,
                        help="Value dictating how much to erode the binary mask"
                             " before selecting a location for noise. The "
                             "default value assumes a slightly generous mask.")
    parser.add_argument("--location", "-l", action="store", type=int, nargs="+",
                        help="Specifies a target location for injecting noise. "
                             "This location must live within the provided mask "
                             "in voxel coordinates. If not provided, a random "
                             "location within the mask will be used.")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Disables checks and restrictions on noise that "
                             "may be not recommended for a typical workflow. By"
                             " default, locations can only be specified within "
                             "the mask, but this overrides that behaviour.")
    parser.add_argument("--mode", "-m", action="store",
                        choices=["single", "uniform", "independent"],
                        default="single",
                        help="Determines where noise will be injected in the "
                             "case of higher-dimensional images than masks. "
                             "'Single' (default) will choose a single position "
                             "in all higher dimensions, resulting in 1 point of"
                             " noise. 'Uniform' will choose a location within "
                             "the mask and apply it uniformly across all other "
                             "dimensions. 'Independent' will generate a random "
                             "location within the mask for each volume in the "
                             "remaining dimensions, and is mutually exclusive "
                             "with providing a location.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Toggles verbose output printing.")
    parser.add_argument("--boutiques", action="store_true",
                        help="Toggles creation of a Boutiques descriptor and "
                             "invocation from the tool and inputs.")

    results = parser.parse_args()

    # Just create the descriptor and exit if we set this flag.
    if results.boutiques:
        make_descriptor(parser, results)
        return 0

    # Grab arguments from parser
    image = results.image_file
    mask = results.mask_file
    output = results.output_directory
    scale = results.scale
    intensity = results.intensity
    erode = results.erode
    location = results.location
    force = results.force
    mode = results.mode
    verb = results.verbose

    # Load nifti images and extract their data
    image_loaded = nib.load(image)
    image_data = image_loaded.get_data()

    mask_loaded = nib.load(mask)
    mask_data = mask_loaded.get_data()

    # Add 1-voxel noise with provided parameters
    output_data, loc = one_voxel_noise(image_data, mask_data, scale=scale,
                                       intensity=intensity, erode=erode,
                                       location=location, force=force,
                                       mode=mode)

    # Save noisy image in the same format as the original
    if image_loaded.header_class == nib.Nifti1Header:
        imtype = nib.Nifti1Image
    elif image_loaded.header_class == nib.Nifti2Header:
        imtype = nib.Nifti2Image
    else:
        raise TypeError("Unrecognized header - only Nifti is supported.")

    # Create and save the output Nifti
    output_loaded = imtype(output_data,
                           header=image_loaded.header,
                           affine=image_loaded.affine)

    # Convert to mm coordinates for the user
    if isinstance(loc, tuple):
        loc = [loc]
    mm_loc = []
    for l in loc:
        tmp_mm = nilimage.coord_transform(l[0], l[1], l[2], image_loaded.affine)
        mm_loc += [tuple(float(tmm) for tmm in tmp_mm)]

    # Create output filename and save image
    bname = op.basename(image).split(".")[0]
    modifier = "_1vox-" + str(uuid.uuid1())[0:8]
    output_file = op.join(output, bname + modifier)

    # Save image to a Nifti file
    nib.save(output_loaded, output_file + ".nii.gz")

    # Save noise information to a JSON file
    with open(output_file + ".json", 'w') as fhandle:
        noisedict = {"voxel_location": loc,
                     "mm_location": mm_loc,
                     "base_image": image,
                     "scale": scale,
                     "intensity": intensity}
        fhandle.write(json.dumps(noisedict, indent=4, sort_keys=True))

    if verb:
        print("Noise added in matrix coordinates at: {0}".format(loc))
        print("Noise added in mm coordinates at: {0}".format(mm_loc))
        print("Image stored in: {0}".format(output_file))


if __name__ == "__main__":
    main()
