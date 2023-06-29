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
    # cmd = "pip freeze | grep sn- | cut -d \'=\' -f1"
    cmd = "pip freeze | egrep \'sn-|sn_\'"
    return cmd


def cmd_install(package, verbose, available_packs, user):
    """
    Function to generate the command to install a list of packages using pip

    Parameters
    ----------
    package : str
        List of packages to install.
    verbose : int
        verbose mode.
    available_packs : array
        array of available (pack,version).
    user : int
        For the user mode.

    Returns
    -------
    list(str)
        List of cmd to execute.

    """

    package = package.split(',')

    if 'sn_pipe' in package and len(package) > 1:
        package = swap(package, 'sn_pipe')

    av_packs = available_packs['packname'].tolist()
    av_packs.append('sn_pipe')
    av_packs.append('all')
    av_packs = swap(av_packs, 'sn_pipe')

    packs = list(set(package) & set(av_packs))
    diff = list(set(package)-set(av_packs))

    if diff:
        print('The following package(s) do not exist', diff)
        print('the list of available packages is \n', av_packs)

    if not packs:
        return []

    if 'sn_pipe' in packs:
        packs = swap(packs, 'sn_pipe')

    if 'all' in packs:
        packs = av_packs
        packs.remove('all')

    vv = ''
    if verbose:
        vv = '-v'

    cmdlist = []

    for pack in packs:
        cmdlist += get_install_list(pack, user)

    return cmdlist


def swap(ll, what):
    """
    Function to swap list values

    Parameters
    ----------
    ll : list(str)
        initial list.
    what : str
        val to set at the position 0.

    Returns
    -------
    ll : list
        new list.

    """

    ip = ll.index(what)
    ll[ip] = ll[0]
    ll[0] = what

    return ll


def get_install_list(package, user):
    """
    Function to get the list of installation to perform

    Parameters
    ----------
    package : str
        package to install.
    user : int
        for --user installation.

    Returns
    -------
    cmdlist : list(str)
        List of cmd to execute for installation.

    """

    add_user = ' '
    if user:
        add_user = ' --user '

    cmdlist = []
    if package == 'sn_pipe':
        cmd = 'pip install{}-r requirements.txt --no-deps'.format(add_user)
        cmdlist.append(cmd)
        version = get_version('sn_tools', available_packs)
        cmd = cmd_install_pack('sn_tools', version, add_user)
        cmdlist.append(cmd)
        # get sn_tools version
    else:
        version = get_version(package, available_packs)
        cmd = cmd_install_pack(package, version, add_user)
        cmdlist.append(cmd)

    return cmdlist


def get_version(pack, packages):
    """
    Function to grab the package version

    Parameters
    ----------
    pack : str
        package name.
    packages : array
        array of (package,version).

    Returns
    -------
    version : str
        package version.

    """

    idx = packages['packname'] == pack
    version = packages[idx]['version'][0]

    return version


def cmd_install_pack(package, version, user):
    """
    Function returning the command to install a package

    Parameters
    ----------
    package : str
        package name.
    version : str
        package version.
    user : str
        user option.

    Returns
    -------
    cmd : TYPE
        DESCRIPTION.

    """

    cmd = 'pip install{}git+https://github.com/lsstdesc/{}.git@{}'.format(
        user,
        package, version)

    return cmd


parser = OptionParser()

parser.add_option("--package", type="str", default='sn_pipe',
                  help="package name to install [%default]")
parser.add_option("--verbose", type=int, default=0,
                  help="verbose mode for pip installation [%default]")
parser.add_option("--action", type="str", default='list',
                  help="action to perform: list, install,\
                  uninstall,list_available [%default]")
parser.add_option("--user", type=int, default=1,
                  help="to set --user in pip install [%default]")

opts, args = parser.parse_args()


pack = opts.package
verbose = opts.verbose
action = opts.action
user = opts.user

available_packs = np.loadtxt('pack_version.txt', dtype={'names': (
    'packname', 'version'), 'formats': ('U15', 'U15')})

if action == 'install':
    cmd = cmd_install(pack, verbose, available_packs, user)
    if cmd is not None:
        for cm in cmd:
            os.system(cm)

if action == 'list':
    os.system(cmd_list())

if action == 'uninstall':
    if pack != 'all':
        pp = pack.split(',')
        for pa in pp:
            os.system(cmd_uninstall(pa))
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
