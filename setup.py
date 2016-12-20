"""obliv: ORAM with variable-size blocks and HIRB data structure."""

from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst')) as f:
    long_description = f.read()


setup(
    name='obliv',

    version='0.0.1',

    description='ORAM with variable-size blocks and HIRB data structure',
    long_description=long_description,

    author='Daniel S. Roche',
    author_email='roche@usna.edu',

    license='Unlicense',

    packages=['obliv'],

    install_requires=['pycrypto', 'paramiko'],
)
