from setuptools import setup
import distutils.cmd
import distutils.log
import os
from setuptools.command.install import install
from pip.req import parse_requirements


class InstallCommand(install):
    """A custom command to install requested package to run Survey Strategy Support Pipeline"""

    description = 'Script to install requested package to run Survey Strategy Support Pipeline'
    user_options = install.user_options + [
        ('package=', None, 'package to install'),
        ('branch=', None, 'git branch to consider')
    ]

    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        install.initialize_options(self)
        self.package = ''
        self.branch = 'dev'

    def finalize_options(self):
        """Post-process options."""

        if self.package:
            if self.package not in ['metrics', 'simulation']:
                print('{} impossible to install'.format(self.package))
        install.finalize_options(self)

    def run(self):
        if self.package == 'metrics':
            cmd = 'pip install --user --process-dependency-links git+https://github.com/lsstdesc/sn_metrics.git@{}'.format(
                self.branch)
            os.system(cmd)
            # parse_requirements() returns generator of pip.req.InstallRequirement objects
            install_reqs = parse_requirements(
                'requirements.txt', session='hack')
            # reqs is a list of requirement
            reqs = [str(ir.req) for ir in install_reqs]
            for req in reqs:
                cmd = 'pip install --user {}'.format(req)
                os.system(cmd)
            for req in ['astropy-healpix']:
                cmd = 'pip install --user --no-deps{}'.format(req)
                os.system(cmd)

        install.run(self)


setup(
    name='sn_pipe',
    version='0.1',
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
