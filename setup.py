from setuptools import setup
import distutils.cmd
import distutils.log
import os
from setuptools.command.install import install
# from pip.req import parse_requirements
from pip._internal.req import parse_requirements
import numpy as np


class InstallCommand(install):
    """A custom command to install requested package to run Survey Strategy Support Pipeline"""

    description = 'Script to install requested package to run Survey Strategy Support Pipeline'
    user_options = install.user_options + [
        ('package=', None, 'package to install')
    ]

    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        install.initialize_options(self)
        self.package = ''
        self.packgs = ['sn_metrics', 'sn_simulation',
                       'sn_fit_lc', 'sn_plotters']
        # load package versions
        self.packvers = np.loadtxt('pack_version.txt', dtype={'names': (
            'packname', 'version'), 'formats': ('U15', 'U15')})

    def finalize_options(self):
        """Post-process options."""

        if self.package:
            if self.package not in ['metrics', 'simulation']:
                print('{} impossible to install'.format(self.package))
        install.finalize_options(self)

    def run(self):
        # install dependencies first
        if self.package == 'sn_pipe':
            cmd = 'pip install --user -r requirements.txt --no-deps'
            os.system(cmd)

        else:
            if self.package != 'all':
                # now install the requested package
                # get the version for this package
                idx = self.packvers['packname'] == self.package
                version = self.packvers[idx]['version'].item()
                cmd = 'pip install -v --user git+https://github.com/lsstdesc/{}.git@{}'.format(
                    self.package, version)
                os.system(cmd)
            else:
                for pack in self.packgs:
                    # get the version for this package
                    idx = self.packvers['packname'] == pack
                    version = self.packvers[idx]['version'].item()
                    cmd = 'pip install --user git+https://github.com/lsstdesc/{}.git@{}'.format(
                        pack, version)
                    os.system(cmd)

        install.run(self)


setup(
    name='sn_pipe',
    version='v1.0.0',
    description='A framework to run the Survey Strategy Support pipeline for supernovae',
    url='http://github.com/lsstdesc/sn_pipe',
    author='Philippe Gris',
    author_email='philippe.gris@clermont.in2p3.fr',
    license='BSD',
    python_requires='>=3.5',
    zip_safe=False,
    cmdclass={
        'install': InstallCommand,
    },

)
