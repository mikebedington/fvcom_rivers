from setuptools import setup

setup(name='fvcom_river',
        version='0.1.0',
        description='Producing river data for FVCOM models',
        url='https://github.com/mikebedington/fvcom_rivers',
        author='mbe',
        author_email='mbe@pml.ac.uk',
        packages=['fvcom_river'],
        install_requires=['numpy', 'sqlite3', 'datetime', 'subprocess', 'gpxpy.geo', 'sklearn', 'tensorflow', 'keras==2.3.1'],
        zip_safe=False)

