#!/usr/bin/env python

from textwrap import dedent
from numpy.distutils.core import setup, Extension

setup(
    name='ttim',
    version='0.22',
    description=dedent('''\
        TTim is a multi-layer transient analytic element solver for modeling
        groundwater flow
    '''),
    author='Mark Bakker',
    author_email='mark.bakker@tudelft.nl',
    url='https://code.google.com/p/ttim/',
    license='MIT License',
    py_modules=['ttim'],
    ext_modules=[
        Extension('bessel', ['bessel.f95']),
        Extension('invlap', ['invlap.f90']),
    ]
)
