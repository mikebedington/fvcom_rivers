from setuptools import setup

setup(name='fvcom_river',
        version='0.1.0',
        description='Producing river data for FVCOM models',
        url='git@gitlab.ecosystem-modelling.pml.ac.uk:mbe/FVCOM_rivers.git',
        author='mbe',
        author_email='mbe@pml.ac.uk',
        packages=['fvcom_river'],
        install_requires=['numpy', 'sqlite3', 'datetime', 'subprocess', 'gpxpy.geo', 'sklearn', 'tensorflow', 'keras'],
        zip_safe=False)

