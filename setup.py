from setuptools import setup

setup(
    name='drought_t',
    version='0.4.1',
    description='Drought temporal analysis.',
    url='https://github.com/rrealrangel/drought_t',
    author='R. A. Real-Rangel',
    author_email='rrealr@iingen.unam.mx',
    license='GPL-3.0',
    packages=[
        'drought_t'
        ],
    install_requires=[
        'dask',
        'matplotlib',
        'netcdf4',
        'numpy',
        'pandas',
        'pathlib2',
        'scipy',
        'toml',
        'toolz',
        'xarray'
        ],
    zip_safe=False
    )
