#!/usr/bin/env python

import nibabel as nib


def image_writer(filename, input_image, output_data):
    if input_image.header_class == nib.Nifti1Header:
        imtype = nib.Nifti1Image
    elif input_image.header_class == nib.Nifti2Header:
        imtype = nib.Nifti2Image
    else:
        raise TypeError("Unrecognized header - only Nifti is supported.")

    # Create and save the output Nifti
    output_loaded = imtype(output_data,
                           header=input_image.header,
                           affine=input_image.affine)

    # Save image to a Nifti file
    nib.save(output_loaded, filename)
    pass
