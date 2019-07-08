#!/usr/bin/env python

import boutiques as bosh
import os.path as op
from argparse import ArgumentParser
import json


def skull_strip(image, mask, fourd=False, debug=False):
    mode = "launch"
    descriptor = "zenodo.1482743"
    invocation = {
        "infile": image,
        "binary_mask_flag": True,
        "maskfile": mask,
        "verbose_flag": True,
        "debug_flag": True,
        "no_seg_output_flag": True,
        "whole_set_mask_flag": fourd
    }

    volumes = ["-v"]
    for fl in [image, mask]:
        volumes += ["{0}:{0}".format(op.abspath(op.dirname(fl)))]

    args = [mode, descriptor, json.dumps(invocation), *volumes]
    if debug:
        args += ['-x']

    bosh.execute(*args)


def one_voxel(image, mask, output, intensity=1.01, scale=True, location=None,
              debug=False):
    mode = "launch"
    descriptor = "zenodo.123456"
    invocation = {
        "image": image,
        "mask": mask,
        "output": output,
        "intensity": intensity,
        "scale": scale,
        "location": location
    }

    volumes = ["-v"]
    for fl in [image, mask, output]:
        volumes += ["{0}:{0}".format(op.abspath(op.dirname(fl)))]

    args = [mode, descriptor, json.dumps(invocation), *volumes]
    if debug:
        args += ['-x']

    bosh.execute(*args)


def make_descriptor(parser, arguments):
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
    parser = ArgumentParser(__file__)
    parser.add_argument("image",
                        help="")
    parser.add_argument("mask",
                        help="")
    parser.add_argument("output",
                        help="")
    parser.add_argument("--debug", "-x", action="store_true")
    parser.add_argument("--fourd", "-F", action="store_true")
    parser.add_argument("--intensity", "-i", action="store", type=float,
                        default=1.01)
    parser.add_argument("--scale", "-s", action="store_true",
                        help="")
    parser.add_argument("--boutiques", action="store_true")
    results = parser.parse_args()

    if results.boutiques:
        make_descriptor()
        return 0

    image = results.image
    mask = results.mask
    output = results.output
    debug = results.debug
    fourd = results.fourd
    intensity = results.intensity
    scale = results.scale

    if not op.isfile(mask):
        skull_strip(image, mask, fourd=fourd, debug=debug)
    one_voxel(image, mask, output, intensity, scale, debug=debug)


if __name__ == "__main__":
    main()
