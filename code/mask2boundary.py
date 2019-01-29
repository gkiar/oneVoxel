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

    results = parser.parse_args()

    mask2boundary(results.mask, results.output, width=results.width)

if __name__ == "__main__":
    main()
