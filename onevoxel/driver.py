#!/usr/bin/env python

from argparse import ArgumentParser
import os.path as op
import json
import uuid
import os

from nilearn import image as nilimage
import nibabel as nib

from .noise import generate_noise_params, apply_noise_params


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


def main(args=None):
    parser = ArgumentParser(__file__,
                            description="Adds noise to a single voxel within an"
                                        " image, conditioned by an image mask.")
    parser.add_argument("image_file",
                        help="Nifti image to be injected with one-voxel noise. "
                             "Default behaviour is that this will be done at a "
                             "random location within an image mask.")
    parser.add_argument("output_directory",
                        help="Path for where the resulting Nifti image with one"
                             " voxel noise will be stored.")
    parser.add_argument("--mask_file", "-m", action="store",
                        help="Nifti image containing a binary mask for the "
                             "input image. The noise location will be selected "
                             "randomly within this mask, unless a location is "
                             "provided.")
    parser.add_argument("--no_scale", "-s", action="store_true",
                        help="Dictates the way in which noise is aplpied to the"
                             " image. If set, the value specified with the "
                             "intensity flag will be set to the new value. If "
                             "not set, the intensity value will be multiplied "
                             "by the original image value at the location.")
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
    parser.add_argument("--mode", action="store",
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
    parser.add_argument("--clean", "-c", action="store_true",
                        help="Deletes the noisy Nifti image from disk. This is "
                             "intended to be used to save space, and the images"
                             " can be regenerated using the 'apply' option and "
                             "providing the associated JSON file.")
    parser.add_argument("--apply_noise", "-a", action="store",
                        help="Provided with a path to 1-voxel noise associated "
                             "JSON file, will apply noise to the image. A hash "
                             "is stored in this file to verify that the same "
                             "noise is injected each time the file is created.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Toggles verbose output printing.")
    parser.add_argument("--boutiques", action="store_true",
                        help="Toggles creation of a Boutiques descriptor and "
                             "invocation from the tool and inputs.")

    results = parser.parse_args() if args is None else parser.parse_args(args)

    # Just create the descriptor and exit if we set this flag.
    if results.boutiques:
        make_descriptor(parser, results)
        return 0

    # Grab arguments from parser
    image = results.image_file
    output = results.output_directory
    clean = results.clean
    apply_noise = results.apply_noise
    verb = results.verbose

    if apply_noise:
        output_file = op.splitext(apply_noise)[0]
        # Handle special case: apply_noise and clean means delete noisy image.
        if clean:
            if op.isfile(output_file + ".nii.gz"):
                os.remove(output_file + ".nii.gz")
            return 0
    else:
        # Create output filename for noise data
        bname = op.basename(image).split(".")[0]
        modifier = "_1vox-" + str(uuid.uuid1())[0:8]
        output_file = op.join(output, bname + modifier)

    # Load nifti images and extract their data
    image_loaded = nib.load(image)
    image_data = image_loaded.get_data()

    # If a noise file is provided, use it to grab noise features
    if apply_noise:
        with open(apply_noise) as fhandle:
            noise_data = json.loads(fhandle.read())

        scale = noise_data['scale']
        intensity = noise_data['intensity']
        loc = [tuple(vl) for vl in noise_data["voxel_location"]]
        mm_loc = noise_data["mm_location"]
        original_hash = noise_data['matrix_hash']

    # If not, generate noise based on parameters from the command-line
    else:
        scale = not results.no_scale
        intensity = results.intensity
        location = results.location
        force = results.force
        mode = results.mode
        mask = results.mask_file
        erode = results.erode
        original_hash = None

        if not mask:
            raise ValueError("Must provide a mask for generating noise.")
        mask_loaded = nib.load(mask)
        mask_data = mask_loaded.get_data()

        # Generate noise based on input params
        loc = generate_noise_params(image_data, mask=mask_data, erode=erode,
                                    location=location, force=force)

    # Apply noise to image
    output_data, output_hash = apply_noise_params(image_data, loc, scale=scale,
                                                  intensity=intensity)

    # Verify that the hashes match for our noisy images.
    if original_hash and output_hash != original_hash:
        print("WARNING: Noisy image hash is different from expected hash.")

    # Only create noise JSON if there wasn't one provided
    if not apply_noise:
        # Get noise locations in mm (useful for visualizing)
        mm_loc = []
        for l in loc:
            tmp_mm = nilimage.coord_transform(l[0], l[1], l[2],
                                              image_loaded.affine)
            mm_loc += [tuple(float(tmm) for tmm in tmp_mm)]

        # Save noise information to a JSON file
        with open(output_file + ".json", 'w') as fhandle:
            noisedict = {"voxel_location": loc,
                         "mm_location": mm_loc,
                         "base_image": image,
                         "matrix_hash": output_hash,
                         "scale": scale,
                         "intensity": intensity}
            fhandle.write(json.dumps(noisedict, indent=4, sort_keys=True))

    if verb:
        print("Noise added in matrix coordinates at: {0}".format(loc))
        print("Noise added in mm coordinates at: {0}".format(mm_loc))
        print("Image stored in: {0}".format(output_file))

    # If we're being clean, return before saving an image.
    if clean:
        return 0

    # If we're not cleaning, save noisy image in the same format as the original
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

    # Save image to a Nifti file
    nib.save(output_loaded, output_file + ".nii.gz")


if __name__ == "__main__":
    main()
