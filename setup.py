from setuptools import setup

# get the version here
pkg_vars = {}

with open("version.py") as fp:
    exec(fp.read(), pkg_vars)

setup(
    name='sn_pipe',
    version=pkg_vars['__version__'],
    description='A framework to run the Survey Strategy Support pipeline for supernovae',
    url='http://github.com/lsstdesc/sn_pipe',
    author='Philippe Gris',
    author_email='philippe.gris@clermont.in2p3.fr',
    license='BSD',
    python_requires='>=3.5',
    zip_safe=False,
    # cmdclass={
    #    'install': InstallCommand,
    # },
    packages=[],

)
