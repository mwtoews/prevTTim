====
TTim
====

:Author: Mark Bakker
:Website: https://code.google.com/p/ttim/

TTim is a multi-layer transient analytic element solver for modeling
groundwater flow.

Dependancies
------------

TTim depends on:

 - Python 2.x
 - numpy
 - scipy
 - matplotlib

Build extensions
----------------

A Fortran 90/95 compiler is requird by f2py to build the Python extensions.

Run the command::

    $ python setup.py build_ext

Or on Windows, specify a compiler, e.g. MinGW::

    $ python.exe setup.py build_ext --fcompiler=gnu95 --compiler=mingw32

Test
----

Build the extensions in-place, then test::

    $ python setup build_ext --inplace
    $ python setup test

Install
-------

Run the command (as root or with ``sudo``)::

    $ python setup install
