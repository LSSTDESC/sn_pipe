from optparse import OptionParser
import yaml
from importlib import import_module
import sn_simu_input as simu
import re
from sn_tools.sn_io import make_dict_from_config,make_dict_from_optparse


def decrypt_parser(parser):
    """
    Method to decrypt the parser help

    Parameters
    ---------------
    parser: optparse parser


    Returns
    ----------
    dict with decrypted infos

    """
    
    file_name = 'help_script.txt'
    file_object  = open(file_name,'w')
    parser.print_help(file_object)
    file_object.close()

    file = open(file_name, 'r') 
    line = file.read().splitlines()

    params = {}
    for i,ll in enumerate(line):
        lolo = ' '.join(ll.split(' ')).split()
        if lolo and lolo[0][:2]=='--':
            key = lolo[0].split('--')[1]
            key = key.split('=')
            params[key[0]]= (key[1].split('/')[0],key[1].split('/')[1])

    return params

def make_dict_old(thedict,key,what,val):
    """
    Method to append to a dict infos
    The goal is in fine to create a yaml file

    Parameters
    --------------
    thedict: dict
      the dict to append to
    key: str
     key used to append
    what: (str,str)
      name and type
    val: str
      value

    Returns
    ----------
    thedict: dict
      resulting dict

    """
      
    
    keym = key.split('_')[0]
    keys = ''
    if '_' in key:
        keys = '_'.join(key.split('_')[1:])

    names = what[0].split(',')
    dtypes = what[1].split(',')
    val = val.split(',')
    
    valb = [val[i] if dtypes[i]=='str' else eval('{}({})'.format(dtypes[i],val[i]))  for i in range(len(dtypes))]
    
    if keys=='':
        if names[0] != 'value':
            thedict[keym]=dict(zip(names,valb))
        else:
            thedict[keym]=valb[0]
    else:
        if keym not in thedict.keys():
            thedict[keym] = {}
        if names[0] != 'value':
            thedict[keym][keys]=dict(zip(names,valb))  
        else:
            thedict[keym][keys]=valb[0] 
            
    return thedict

path = simu.__path__

confDict = make_dict_from_config(path[0],'config_simulation.txt')

parser = OptionParser()
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0],vals[1]))
    parser.add_option('--{}'.format(key),help='{} [%default]'.format(vals[2]),default=vv,type=vals[0],metavar='')

parser.add_option('--fileName',help='output file name [%default]',default='config.yaml',type='str')

opts, args = parser.parse_args()

newDict = {}
for key, vals in confDict.items():
    newval = eval('opts.{}'.format(key))
    newDict[key]=(vals[0],newval)


dd = make_dict_from_optparse(newDict)

print('config',dd)
with open(opts.fileName, 'w') as f:
    data = yaml.dump(dd, f)
