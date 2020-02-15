import os
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--package", type="str", default='metrics',
                  help="package name to install [%default]")
parser.add_option("--gitbranch", type="str", default='dev',
                  help="gitbranch of the package [%default]")

opts, args = parser.parse_args()


pack = opts.package
gitbranch = opts.gitbranch

cmd = "pip install . --user --install-option=\"--package={}\" --install-option=\"--branch={}\"".format(
    pack, gitbranch)

print(cmd)
os.system(cmd)
