from setuptools import setup
import distutils.cmd
import distutils.log
import os
from setuptools.command.install import install


class InstallCommand(install):
    """A custom command to install requested package to run Survey Strategy Support Pipeline"""

    description = 'Script to install requested package to run Survey Strategy Support Pipeline'
    user_options = install.user_options+[
        # The format is (long option, short option, description).
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
        install.finalize_options(self)
        if self.package:
            if self.package not in ['metrics', 'simulation']:
                print('{} impossible to install'.format(self.package))

    def run(self):
        if self.package == 'metrics':
            cmd = 'pip install --user --process-dependency-links git+https://github.com/lsstdesc/sn_metrics.git@{}'.format(
                self.branch)
            os.system(cmd)
        install.run(self)
        """Run command."""
        """
        command = ['source']
        if self.release:
            command.append('%s' % self.release)
            # command.append(os.getcwd())
            # command.append('setup lsst_sims')
            self.announce(
                'Running command: %s' % str(command),
                level=distutils.log.INFO)
            subprocess.check_call(command)
        """


setup(
    name='sn_pipe',
    version='0.1',
    description='Metrics for supernovae',
    url='http://github.com/lsstdesc/sn_pipe',
    author='Philippe Gris',
    author_email='philippe.gris@clermont.in2p3.fr',
    license='BSD',
    python_requires='>=3.5',
    zip_safe=False,
    # install_requires=[
    #    'h5py==2.7.1'
    # ],
    cmdclass={
        'install': InstallCommand,
    },

)
