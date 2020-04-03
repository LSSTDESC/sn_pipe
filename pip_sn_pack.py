import os
import subprocess
from optparse import OptionParser


def cmd_uninstall(pack):
    """
    Function to generate the command to uinstall a package using pip

    Parameters
    --------------
    pack: str
      name of the package to uninstall

    Returns
    -----------
    cmd: str
      cmd to apply
    """
    cmd = "pip uninstall {}".format(pack)
    return cmd


def cmd_list():
    """
    Function to to generate the command giving the list of  packages installed with pip

    Returns
    -----------
    cmd: str
      cmd to apply
    """
    cmd = "pip freeze | grep sn- | cut -d \'=\' -f1"
    return cmd


def cmd_install(package, gitbranch):
    """
    Function to generate the command to install a package using pip

    Parameters
    --------------
    pack: str
      name of the package to install
    gitbranch: str
        git branch (master, dev, ...) of the package to install

    Returns
    -----------
    cmd: str
      cmd to apply
    """
    cmd = "pip install . --user --install-option=\"--package={}\" --install-option=\"--branch={}\"".format(
        pack, gitbranch)
    return cmd


parser = OptionParser()

parser.add_option("--package", type="str", default='metrics',
                  help="package name to install [%default]")
parser.add_option("--gitbranch", type="str", default='dev',
                  help="gitbranch of the package [%default]")
parser.add_option("--action", type="str", default='list',
                  help="action to perform: list, install, uninstall [%default]")

opts, args = parser.parse_args()


pack = opts.package
gitbranch = opts.gitbranch
action = opts.action

if action == 'install':
    os.system(cmd_install(pack, gitbranch))

if action == 'list':
    os.system(cmd_list())

if action == 'uninstall':
    if pack != 'all':
        os.system(cmd_uninstall(pack))
    else:
        # this will uninstall the entire pipeline
        # get all the packages
        packgs = subprocess.Popen(
            cmd_list(), shell=True, stdout=subprocess.PIPE).stdout.read()
        listpk = packgs.decode().split('\n')
        for pp in listpk:
            if pp != '':
                os.system(cmd_uninstall(pp))
