#!/usr/bin/env python

import boutiques as bosh
import os.path as op
from argparse import ArgumentParser
import json


def skull_strip(image, output, fourd=False, debug=False):
    mode = "launch"
    descriptor = "zenodo.1482743"
    invocation = {
        "infile": image,
        "binary_mask_flag": True,
        "maskfile": output,
        "verbose_flag": True,
        "debug_flag": True,
        "no_seg_output_flag": True,
        "whole_set_mask_flag": fourd
    }

    volumes = ["-v"]
    for fl in [image, output]:
        volumes += ["{0}:{0}".format(op.abspath(op.dirname(fl)))]

    args = [mode, descriptor, json.dumps(invocation), *volumes]
    if debug:
        args += ['-x']

    bosh.execute(*args)


def main():
    parser = ArgumentParser()
    parser.add_argument("image",
                        help="")
    parser.add_argument("output",
                        help="")
    parser.add_argument("--debug", "-x", action="store_true")
    parser.add_argument("--fourd", "-F", action="store_true")
    results = parser.parse_args()

    image = results.image
    output = results.output
    debug = results.debug
    fourd = results.fourd

    skull_strip(image, output, fourd=fourd, debug=debug)


if __name__ == "__main__":
    main()
