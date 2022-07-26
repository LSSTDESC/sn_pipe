import os
import subprocess
from optparse import OptionParser
import numpy as np


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
    #cmd = "pip freeze | grep sn- | cut -d \'=\' -f1"
    cmd = "pip freeze | egrep \'sn-|sn_\'"
    return cmd


def cmd_install(package, verbose, available_packs):
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

    if package not in available_packs['packname'].tolist() and package != 'sn_pipe' and package != 'all':
        print('The package you are trying to install does not exist')
        print('The list of available packages is ',
              available_packs['packname'].tolist())
        return None

    vv = ''
    if verbose:
        vv = '-v'

    cmd = "pip {} install . --user --install-option=\"--package={}\"".format(vv,
                                                                             package)

    print('install cmd', cmd)
    return cmd


parser = OptionParser()

parser.add_option("--package", type="str", default='sn_pipe',
                  help="package name to install [%default]")
parser.add_option("--verbose", type=int, default=0,
                  help="verbose mode for pip installation [%default]")
parser.add_option("--action", type="str", default='list',
                  help="action to perform: list, install, uninstall,list_available [%default]")

opts, args = parser.parse_args()


pack = opts.package
verbose = opts.verbose
action = opts.action
available_packs = np.loadtxt('pack_version.txt', dtype={'names': (
    'packname', 'version'), 'formats': ('U15', 'U15')})

if action == 'install':
    cmd = cmd_install(pack, verbose, available_packs)
    if cmd is not None:
        os.system(cmd)

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
                os.system(cmd_uninstall(pp.split(' ')[0]))

if action == 'list_available':
    print('The list of available packages is ',
          available_packs['packname'].tolist())
