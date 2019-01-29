#!/usr/bin/env python

from argparse import ArgumentParser
import nibabel as nib
import numpy as np
from scipy import ndimage
from nilearn import image as nilimage


def one_voxel_noise(image, mask, output, scale=True, intensity=0.01, erode=3,
                    location=[], force=False):
    image_loaded = nib.load(image)
    mask_loaded = nib.load(mask)

    image_data = image_loaded.get_data()
    mask_data = mask_loaded.get_data()

    mask_data = ndimage.binary_erosion(mask_data, iterations=int(erode))
    mask_locs = np.where(mask_data > 0)

    if location:
        location = tuple(int(l) for l in location)
        # TODO: enforce location is within the mask
    else:
        index = np.random.randint(0, high=len(mask_locs[0]))
        location = tuple(ml[index] for ml in mask_locs)

    if len(location) > 3 and not force:
        location = location[0:3]

    if scale:
        image_data[location] = image_data[location]*(1 + intensity)
    else:
        image_data[location] = intensity

    if image_loaded.header_class == nib.Nifti1Header:
        func = nib.Nifti1Image
    elif image_loaded.header_class == nib.Nifti2Header:
        func = nib.Nifti2Image
    else:
        print("What kind of image is this...?")
        print(image_loaded.header_class)
        return -1

    output_loaded = func(image_data,
                         header=image_loaded.header,
                         affine=image_loaded.affine)
    nib.save(output_loaded, output)

    # Add one to make it 1-indexed
    location = tuple(l+1 for l in location)
    real_location = nilimage.coord_transform(location[0], location[1],
                                             location[2], image_loaded.affine)
    print(location, real_location)
    return location


def make_descriptor(parser, arguments=None):
    import boutiques.creator as bc
    import os.path as op
    import json

    desc = bc.CreateDescriptor(parser, execname=op.basename(__file__))
    basename = op.splitext(__file__)[0]
    desc.save(basename + ".json")

    if arguments is not None:
        invo = desc.createInvocation(arguments)
        invo.pop("boutiques")

        with open(basename + "_inputs.json", "w") as fhandle:
            fhandle.write(json.dumps(invo, indent=4))


def main():
    parser = ArgumentParser(__file__)
    parser.add_argument("image_file",
                        help="Nifti image to be injected with one-voxel noise. "
                             "Default behaviour is that this will be done at a "
                             "random location within an image mask.")
    parser.add_argument("mask_file",
                        help="Nifti image containing a binary mask for the "
                             "input image. The noise location will be selected "
                             "randomly within this mask, unless a location is "
                             "provided.")
    parser.add_argument("output_file",
                        help="Nifti image for the perturbed image with one "
                             "voxel noise. If you want to edit in place, "
                             "provide the name of the image_file again here.")
    parser.add_argument("--scale", "-s", action="store_true",
                        help="Dictates the way in which noise is aplpied to the"
                             " image. If not set, the value specified with the "
                             "intensity flag will be set to the new value. If "
                             "set, the intensity value will be multiplied by "
                             "the original image value at the target location.")
    parser.add_argument("--intensity", "-i", action="store", type=float,
                        default=1.01,
                        help="The intensity of the noise to be injected in the "
                             "image. Default value is 1.01 so specifying the "
                             "scale flag alone will result in a 1% intensity "
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
                             " default, locations can only be specified in the "
                             "first 3 dimensions and will be applied uniformly "
                             "across the rest - this option and a higher "
                             "dimensional mask or location enables injection at"
                             "specific points in higher dimensions.")
    parser.add_argument("--boutiques", action="store_true",
                        help="Toggles creation of a Boutiques descriptor and "
                             "invocation from the tool and inputs.")

    results = parser.parse_args()

    if results.boutiques:
        make_descriptor(parser, results)
        return 0

    image = results.image
    mask = results.mask
    output = results.output
    scale = results.scale
    intensity = results.intensity
    erode = results.erode
    location = results.location
    force = results.force

    one_voxel_noise(image, mask, output, scale=scale, intensity=intensity,
                    erode=erode, location=location, force=force)


if __name__ == "__main__":
    main()
