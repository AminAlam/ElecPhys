import os
from setuptools import setup, find_packages
from distutils.core import Extension
from pathlib import Path

MINIMAL_DESCRIPTION = '''ElecPhys: A Python package for electrophysiology data analysis. It provides tools for data loading, analysis, conversion, preprocessing, and visualization.'''

def get_requires():
    """Read requirements.txt."""
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    try:
        with open(requirements_file, "r") as f:
            requirements = f.read()
        return list(filter(lambda x: x != "", requirements.split()))
    except FileNotFoundError:
        return []

def read_description():
    """Read README.md and CHANGELOG.md."""
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path) as r:
            description = "\n" + r.read()
        return description
    return MINIMAL_DESCRIPTION

setup(
    name="ElecPhys",
    version="0.0.55",
    author='Amin Alam',
    description='Electrophysiology data processing',
    long_description=read_description(),
    long_description_content_type='text/markdown',
    install_requires=get_requires(),
    python_requires='>=3.8',
    license='MIT',
    url='https://github.com/AminAlam/ElecPhys',
    keywords=['EEG', 'signal-processing', 'electrophysiology', 'data-analysis', 'data-visualization', 'data-conversion', 'data-preprocessing', 'data-loading', 'rhd', 'notch-filter', 'dft', 'stft', 'fourier-transform'],
    entry_points={
        'console_scripts': [
            'elecphys=elecphys.main:main',
        ],
    },
    packages=['elecphys'],
    package_data={'elecphys': ['*.m', 'elecphys/matlab_scripts/*.m']},
    include_package_data=True,
)
