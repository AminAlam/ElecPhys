import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.core import Extension

MINIMAL_DESCRIPTION = '''ElecPhys: A Python package for electrophysiology data analysis. It provides tools for data loading, analysis, conversion, preprocessing, and visualization.'''

def get_requires():
    """Read requirements.txt."""
    requirements = open('requirements.txt', "r").read()
    return list(filter(lambda x: x != "", requirements.split()))

def read_description():
    """Read README.md and CHANGELOG.md."""
    try:
        with open("README.md") as r:
            description = "\n"
            description += r.read()
        return description
    except Exception:
        return MINIMAL_DESCRIPTION
    


setup(
    name="ElecPhys",
    version="0.0.1",
    author='Amin Alam',
    description='Electrophysiology data processing',
    long_description=read_description(),
    long_description_content_type='text/markdown',
    install_requires=get_requires(),
    python_requires='>=3.8',
    license='MIT',
    include_package_data=True,
    url='https://github.com/AminAlam/ElecPhys',
    keywords="EEG python singal-processing electrophysiology",
    entry_points={
        'console_scripts': [
            'ElecPhys=src.main:main',
        ]
    },
    # specify the name of the compiled extension as cpp_backend
    packages=['ElecPhys']
    )
