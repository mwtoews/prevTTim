#!/usr/bin/env python
# To use:
#	   python setup.py build
#	   python setup.py install
#	   python setup.py install --prefix=...
#	   python setup.py bdist --format=wininst
#	   python setup.py bdist --format=rpm
#	   python setup.py sdist --formats=gztar,zip

# This setup script authored by Philippe Le Grand, June 13th 2005

import sys

if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'alpha',0):
	raise SystemExit, "Python 2.6 or later required to build TTim"


from distutils.core import setup, Extension
from bdist_mpkg import *

setup (name = "TTim",
	   extra_path = 'TTim',
	   version = "0.01.py26",
	   author="Mark Bakker",
	   author_email="mark.bakker@tudelft.nl",
	   py_modules = ["ttim","ttimtest1","ttimtest2","ttimtest3","ttimtest4"],
# This trick might be original; I haven't found it anywhere.
# The precompiled Fortran library is passed as a data file,
# so that dist does not try and recompile on the destination machine
       data_files = [("Lib/site-packages/ttim",["bessel.pyd","invlap.pyd"])]
#	   ext_modules= [Extension("besselaes",["besselaes.f90","trianglemodule.c"])]
	   )
