#!/usr/bin/env python
# To use:
#	   python setup.py build
#	   python setup.py install
#	   python setup.py install --prefix=...
#	   python setup.py bdist --format=wininst
#	   python setup.py bdist --format=rpm
#	   python setup.py sdist --formats=gztar,zip
#
# To create Mac installer:
# Install py2app
# At terminal prompt:
# /Library/Frameworks/Python.framework/Versions/7.2/bin/bdist_mpkg  # Last known location
# bdist_mpkg setup.py build
# Installer assumes that sitepackages are installed in sys.prefix+'/lib/python2.pyver/site-packages/ttim as is the case in EPD
# pyver needs to be set to the correct value

import sys

#if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'alpha',0):
#	raise SystemExit, "Python 2.6 or later required to build TTim"


from distutils.core import setup, Extension

# Specify Python version 2.pyver:
pyver = 7

setup (name = "ttim",
	   extra_path = 'ttim',
	   version = "0.2",
	   author="Mark Bakker",
	   author_email="mark.bakker@tudelft.nl",
	   py_modules = ["ttim"],
# This trick might be original; I haven't found it anywhere.
# The precompiled Fortran library is passed as a data file,
# so that dist does not try and recompile on the destination machine
       data_files = [(sys.prefix+'/lib/python2.'+str(pyver)+'/site-packages/ttim',["bessel.so","invlap.so",
                      "x0y0.fth","x1y0.fth","x5y0.fth","x10y0.fth",
                      "CCrack.dat","NCrack.dat","SCrack.dat","Crackxst10.dat",
                      "ttimtest1.py","ttimtest2.py","ttimtest3.py","ttimtest4.py","neuman.png"])]
#	   ext_modules= [Extension("besselaes",["besselaes.f90","trianglemodule.c"])]
	   )
