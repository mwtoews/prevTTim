#!/usr/bin/env python
import os
import sys
from glob import glob
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

ext_args = {}
if os.sys.platform.startswith('win'):
    ext_args['extra_link_args'] = ['-static']

description = '''\
A multi-layer transient analytic element solver for modeling groundwater flow
'''

long_description = '''\
TTim (pronounce "Tee Tim") is a solver for transient multi-aquifer flow based
on the Laplace-transform analytic element method and is developed at the Delft
University of Technology in Delft, The Netherlands.
'''

classifier_txt = '''\
Development Status :: 5 - Production/Stable
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Fortran
Topic :: Scientific/Engineering
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
'''

setup(
    name = 'TTim',
    version = '0.22',
    author = 'Mark Bakker',
    author_email = 'mark.bakker@tudelft.nl',
    url = 'http://code.google.com/p/ttim/',
    download_url = 'http://code.google.com/p/ttim/downloads/list',
    description = description,
    long_description = long_description,
    classifiers = [x for x in classifier_txt.split("\n") if x],
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
    # Fortran extensions are built with f2py
    ext_modules = [
        Extension('ttim.bessel', ['ttim/bessel.f95'], **ext_args),
        Extension('ttim.invlap', ['ttim/invlap.f90'], **ext_args),
    ],
    # python setup.py test
    cmdclass = {'test': test},
)
