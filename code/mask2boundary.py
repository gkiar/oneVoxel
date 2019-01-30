#!/usr/bin/env python

from argparse import ArgumentParser
from scipy import ndimage
import nibabel as nib
import numpy as np


def mask2boundary(mask, output, width=4):
    mask_loaded = nib.load(mask)
    mask_data = mask_loaded.get_data()

    outer = ndimage.binary_dilation(mask_data,
                                    iterations=int(np.floor(width/2))).astype(int)
    inner = ndimage.binary_erosion(mask_data,
                                   iterations=int(np.ceil(width/2))).astype(int)

    mask_data = outer-inner

    if mask_loaded.header_class == nib.Nifti1Header:
        func = nib.Nifti1Image
    elif mask_loaded.header_class == nib.Nifti2Header:
        func = nib.Nifti2Image
    else:
        print("What kind of image is this...?")
        print(mask_loaded.header_class)
        return -1

    output_loaded = func(mask_data,
                         header=mask_loaded.header,
                         affine=mask_loaded.affine)
    nib.save(output_loaded, output)


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
    parser = ArgumentParser(__name__)
    parser.add_argument("mask",
                        help="Nifti image containing a binary mask.")
    parser.add_argument("output",
                        help="Path for output Nifti image containing the mask "
                             "boundary.")
    parser.add_argument("--width", "-w", action="store", type=int,
                        default=4,
                        help="Width of the boundary to be stored.")
    parser.add_argument("--boutiques", action="store_true",
                        help="Toggles creation of a Boutiques descriptor and "
                             "invocation from the tool and inputs.")

    results = parser.parse_args()

    if results.boutiques:
        make_descriptor(parser, results)
        return 0

    mask2boundary(results.mask, results.output, width=results.width)

if __name__ == "__main__":
    main()
