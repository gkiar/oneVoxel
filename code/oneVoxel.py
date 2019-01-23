#!/usr/bin/env python


from argparse import ArgumentParser
import nibabel as nib
import numpy as np
from scipy import ndimage
from nilearn import image as nilimage


def one_voxel_noise(image, mask, output, intensity=0.01, scale=True,
                    location=None, erode=3, debug=False):
    image_loaded = nib.load(image)
    mask_loaded = nib.load(mask)

    image_data = image_loaded.get_data()
    mask_data = mask_loaded.get_data()

    mask_data = ndimage.binary_erosion(mask_data, iterations=int(erode))
    mask_locs = np.where(mask_data > 0)

    if location:
        location = tuple(int(l) for l in location)
    else:
        index = np.random.randint(0, high=len(mask_locs[0]))
        location = tuple(ml[index] for ml in mask_locs)

    if len(location) > 3:
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


def make_descriptor():
    import boutiques.creator as bc
    import os.path as op
    import json

    desc = bc.CreateDescriptor(parser, execname=op.basename(__file__))
    basename = op.splitext(__file__)[0]
    desc.save(basename + ".json")
    invo = desc.createInvocation(arguments)
    invo.pop("boutiques")

    with open(basename + "_inputs.json", "w") as fhandle:
        fhandle.write(json.dumps(invo, indent=4))


def main():
    parser = ArgumentParser()
    parser.add_argument("image",
                        help="")
    parser.add_argument("mask",
                        help="")
    parser.add_argument("output",
                        help="")
    parser.add_argument("--debug", "-x", action="store_true")
    parser.add_argument("--intensity", "-i", action="store", type=float,
                        default=1.01)
    parser.add_argument("--scale", "-s", action="store_true",
                        help="")
    parser.add_argument("--erode", "-e", action="store", type=int,
                        default=3, help="")
    parser.add_argument("--boutiques", action="store_true")

    results = parser.parse_args()

    if results.boutiques:
        make_descriptor()
        return 0

    image = results.image
    mask = results.mask
    output = results.output
    debug = results.debug
    intensity = results.intensity
    scale = results.scale
    erode = results.erode

    one_voxel_noise(image, mask, output, intensity=intensity,
                    scale=scale, erode=erode, debug=debug)

if __name__ == "__main__":
    main()
