#!/usr/bin/env python

import os
import sys
from glob import glob
from textwrap import dedent
from unittest import TextTestRunner, TestLoader
from numpy.distutils.core import Extension, Command, setup


class test(Command):
    """run unit tests after in-place build"""
    description = __doc__
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            import ttim
        except ImportError:
            self.run_command('build_ext')
        from ttim.tests import ttim_testsuite
        tests = TestLoader().loadTestsFromModule(ttim_testsuite)
        t = TextTestRunner(verbosity=2)
        t.run(tests)

setup(
    name = 'TTim',
    version = '0.22',
    description = dedent('''\
        TTim is a multi-layer transient analytic element solver for modeling
        groundwater flow
    '''),
    author = 'Mark Bakker',
    author_email = 'mark.bakker@tudelft.nl',
    url = 'https://code.google.com/p/ttim/',
    license = 'MIT License',
    requires = ['numpy', 'scipy'],
    packages = [
        'ttim',
        'ttim.tests',
        'ttim.examples',
    ],
    package_data = {
        'ttim': [
            'examples/*.dat',
            'examples/*.fth',
    ]},
    # Fortran extensions are built with NumPy's f2py
    ext_modules = [
        Extension('ttim.bessel', ['ttim/bessel.f95']),
        Extension('ttim.invlap', ['ttim/invlap.f90']),
    ],
    # python setup.py test
    cmdclass = {'test': test},
)
