import os
from distutils.core import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='grnf',
    version='0.2.0',
    author='Daniele Zambon',
    author_email='daniele.zambon@usi.ch',
    description=('Graph Random Neural Features.'),
    long_description=read('README.md'),
    packages=['grnf', 'grnf.tf', 'grnf.torch']
)
