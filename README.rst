====
TTim
====

:Author: Mark Bakker
:Website: http://code.google.com/p/ttim/

TTim is a multi-layer transient analytic element solver for modeling
groundwater flow [1]_.

Dependencies
------------

TTim depends on:

 - Python >=2.6, <3
 - numpy
 - scipy
 - matplotlib

Build extensions
----------------

A Fortran 90/95 compiler is required by f2py to build the Python extensions.

Run the command::

    $ python setup.py build_ext

Or on Microsoft Windows, specify a compiler, e.g. MinGW::

    > python.exe setup.py build_ext --fcompiler=gnu95 --compiler=mingw32

Test
----

Build the extensions in-place, then test::

    $ python setup.py build_ext --inplace
    $ python setup.py test

Install
-------

Run the command (as root or with ``sudo``)::

    $ python setup.py install

Distribution
------------

To make a source distribution::

    $ python setup.py sdist

To make a Microsoft Windows installer::

    > python setup.py bdist_wininst

Reference
---------

.. [1] M. Bakker, 2013, Semi-analytic modeling of transient multi-layer
       flow with TTim, Hydrogeology Journal, 21(4), 935-943.
       doi:10.1007/s10040-013-0975-2

