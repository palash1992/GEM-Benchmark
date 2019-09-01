#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import os

ver_file = os.path.join('gemben', '_version.py')

with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'gemben'

INSTALL_REQUIRES = [i.strip() for i in open("requirements.txt").readlines()]

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = __version__

setuptools.setup(
    name='gemben',
    version=VERSION,
    author="Palash Goyal, Di Huang, Ankita Goswami, Sujit Rokka Chhetri, Arquimedes Canedo and Emilio Ferrara",
    author_email="palashgo@usc.edu",
    description="Benchmark for Graph Embedding Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sujit-O/gemben.git",
    packages=setuptools.find_packages(exclude=['dataset', 'venv', 'build', 'dist', 'gemben.egg-info']),
    package_dir={DISTNAME: 'gemben'},
    setup_requires=['sphinx>=2.1.2', 'cmake>=3.12.0'],
    extras_require ={'networkit':["cmake>=3.12.0"]},
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)