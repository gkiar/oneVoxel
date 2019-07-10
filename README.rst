oneVoxel
=========

Small library for adding well described noise to images, with a command-line
utility specifically for application on Nifti formatted images.


Installation
------------

Simple! Just open your favourite terminal and type:

::

    $ pip install onevox


Alongside installing the oneVoxel package, this will also ensure the
dependencies are installed: ``numpy``, ``scipy``, ``nibabel``, and ``nilearn``.

For building this in Docker, you can run the following command:

::

    $ docker build -t gkiar/onevox:local --network host .


Usage
-----

From within Python this library can used to apply noise to arbitrary images:

::

    [1]: import numpy as np
    [2]: np.random.seed(1234)
    [3]: data = np.random.random((10,10,10))  # Create data matrix
    [4]: mask = data[:,:,0] > 0.4  # Define mask as values higher than 0.4 in the first 2D slice

    [5]: from onevoxel import noise  # Load noise utils

    [6]: # Generate noise locations from image and mask 
    [7]: loc = noise.generate_noise_params(data, mask, erode=0, mode='independent')
    [8]: loc
    [(3, 6, 0),
     (4, 4, 1),
     (9, 0, 2),
     (7, 9, 3),
     (1, 2, 4),
     (3, 1, 5),
     (0, 7, 6),
     (9, 0, 7),
     (3, 4, 8),
     (1, 9, 9)]

    [9]: # Apply noise to the image and verify it's in the right spot
    [10]: noisy_data, noisy_hash = noise.apply_noise_params(data, loc, scale=True, intensity=0.01)
    [11]: sorted(list(zip(*np.where(noisy_data != data))), key=lambda elem: elem[2]) == loc
    True


Contributing
------------

Excited by the project and want to get involved?! *Please* check out our
`contributing guide <./CONTRIBUTING.md>`__, and look through the
`issues <https://github.com/gkiar/onevoxel/issues/>`__ to start seeing where
you can lend a hand. We look forward to approving your amazing contributions!
