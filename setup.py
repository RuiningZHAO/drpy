from setuptools import setup
from setuptools import find_packages


VERSION = '0.0.2'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='drspy', 
    version=VERSION, 
    description='A Data Reduction Toolkit for Spectroscopy and Photometry', 
    long_description=LONG_DESCRIPTION, 
    long_description_content_type='text/markdown', 
    packages=find_packages('src'),
    package_dir={'': 'src'},
    zip_safe=False, 
)