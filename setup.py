# coding=utf-8

from distutils.core import setup

setup(
    # Application name:
    name="morghulis",

    # Version number (initial):
    version="0.1.0",

    # Application author details:
    author="Fábio Uechi",
    author_email="fabio.uechi@gmail.com",

    # Packages
    packages=[
        "morghulis",
        "morghulis.widerface",
        "morghulis.afw",
        "morghulis.pascal_faces",
        "morghulis.exporters",
        "morghulis.fddb",
        "morghulis.mafa",
        "morghulis.caltech_faces"
    ],

    # package_dir={'': 'lib'},

    # Include additional files into the package
    include_package_data=True,

    # Details
    url="http://pypi.python.org/pypi/morghulis/",

    #
    # license="LICENSE.txt",
    description="Useful towel-related stuff.",

    # long_description=open("README.txt").read(),

    # Dependent packages (distributions)
    install_requires=[
        'requests'
    ],

    extras_require={
        'tensorflow_export': ["tensorflow"],
        'caffe_export': ["caffe"],
    }
)