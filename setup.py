#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from setuptools import setup

PACKAGENAME = "molecular_columns"
DESCRIPTION = 'Column density calculations of molecular especies'
AUTHOR = 'Jaime E. Pineda'
AUTHOR_EMAIL = 'jpineda@mpe.mpg.de'
LICENSE = "MIT"
URL = "https://github.com/jpinedaf/molecular_columns.git"

# VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
VERSION = '0.4.dev3'

here = os.path.abspath(os.path.dirname(__file__))
def read(fname):
    return open(os.path.join(here, fname)).read()

setup(
    name=PACKAGENAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read('README.md'),
    packages=['molecular_columns'],
    install_requires=['astropy>=5.0', 'numpy'],
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    )
