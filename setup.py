from setuptools import setup

VERSION = "0.3.0pre3"
DEPS = [
        "numpy",
        "scipy",
        "nibabel",
        "nilearn",
        "scikit-learn"
       ]

setup(name="onevox",
      version=VERSION,
      description="Library for adding well described noise to images.",
      long_description=open("./README.rst").read(),
      url="http://github.com/gkiar/onevoxel",
      author="Gregory Kiar",
      author_email="gkiar07@gmail.com",
      classifiers=[
                "Programming Language :: Python",
                "Programming Language :: Python :: 2",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 2.6",
                "Programming Language :: Python :: 2.7",
                "Programming Language :: Python :: 3.4",
                "Programming Language :: Python :: 3.5",
                "Programming Language :: Python :: 3.6",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: Implementation :: PyPy",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                "Natural Language :: English"
                  ],
      license="MIT",
      packages=["onevox"],
      include_package_data=True,
      test_suite="pytest",
      tests_require=["pytest"],
      setup_requires=DEPS,
      install_requires=DEPS,
      entry_points={
        "console_scripts": [
            "onevox=onevox.cli:main",
        ]
      },
      zip_safe=False)
