import numpy as np
import os
import h5py
from astropy.table import Table, Column, vstack
from optparse import OptionParser
import glob
import multiprocessing
import time

def Replace_input(input_file, x1, color, z, prodid, fakename,output_file):
    with open(input_file, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('zval', str(z))
    filedata = filedata.replace('prodid', prodid)
    filedata = filedata.replace('fakename', fakename)
    filedata = filedata.replace('x1val', str(x1))
    filedata = filedata.replace('colorval', str(color))

    with open(output_file, 'w') as file:
        file.write(filedata)


def Replace_fake(input_file, duration, mjd_min, cad, output_file):
    with open(input_file, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('duration', str(duration))
    filedata = filedata.replace('mymin', str(mjd_min))
    filedata = filedata.replace('cad', str(cad))
    with open(output_file, 'w') as file:
        file.write(filedata)


def Process_LC(fname, lc_out, ilc, x1, color):
#def Process_LC(fname, lc_out, x1, color):
    f = h5py.File(fname, 'r')
    keys = list(f.keys())

    print(keys)
    for i in range(0, len(keys), 9):
        #ilc = int(keys[i].split('_')[1])
        #print('hello ilc', i, ilc)
        tab = Table.read(f, path=keys[i])
        table_new = Table(tab)
        #print(tab.dtype)
        
        table_new.add_column(
            Column([tab.meta['daymax']]*len(tab), name='daymax'))
        table_new.add_column(Column([tab.meta['z']]*len(tab), name='z'))
        #print('go')
        for j, val in enumerate(['x0', 'x1', 'color','daymax']):
            ia = i+2*j+1
            ib = ia+1
            tablea = Table.read(f, path=keys[ia])
            tableb = Table.read(f, path=keys[ib])
            #print(tablea.meta)
            #print(tableb.meta)
            epsilon = tablea.meta['epsilon_'+val]
            col = (tablea['flux']-tableb['flux'])/(2.*epsilon)
            #print(tablea.meta)
            #print(tableb.meta)
            table_new.add_column(Column(col, name='d'+val))
        bands = [b[-1] for b in table_new['band']]
        table_new.remove_column('band')
        table_new.add_column(
            Column(bands, name='band', dtype=h5py.special_dtype(vlen=str)))
        #phases = (table_new['time']-table_new['DayMax'])/(1.+table_new['z'])
        #table_new.add_column(Column(phases, name='phase'))
        table_new.add_column(Column([x1]*len(table_new), name='x1'))
        table_new.add_column(Column([color]*len(table_new), name='color'))
        #table_new.write(lc_out, path='lc_'+str(ilc),
        table_new.write(lc_out, path='lc_'+str(ilc),
                        append=True, compression=True)
        #os.system('rm '+fname)


def Run(input_orig, cad_orig, zval,x1, color,outdir,ilc):
    mjd_min = -21.*(1.+zval)
    mjd_max = 63.*(1.+zval)
    duration = (mjd_max-mjd_min)
    #cad = 0.1
    cad = 0.1*(1.+zval)
    #cad = 0.5*(1.+zval)
    prodid = 'Fake_{}_{}_{}'.format(zval,x1,color)
    fake_name = 'input_prod/Fake_cadence_{}_{}_{}.yaml'.format(x1,color,zval)
    input_fakes = 'input_prod/input_fakes_{}_{}_{}.yaml'.format(x1,color,zval)
    Replace_input(input_orig, x1, color, zval, prodid, fake_name,input_fakes)
    Replace_fake(cad_orig, duration, mjd_min, cad, fake_name)

    cmd = 'python run_scripts/simulation/run_simulation.py '+input_fakes
    os.system(cmd)
    #Process_LC(outdir+'/LC_'+prodid+'.hdf5', lc_out, ilc, x1, color)

def zLoop(input_orig, cad_orig, zval, x1, color, outdir, inum,j=0, output_q=None):
    
    for z in zval:
        inum += 1
        print('processing',z)
        Run(input_orig, cad_orig, z, x1, color, outdir,inum)

    print('done',j)
    if output_q is not None:
        return output_q.put({j: 1})

#for ilc, zval in enumerate(np.arange(0.01, 0.9, 0.01)):
#for ilc, zval in enumerate([0.01]):
    # for ilc, zval in enumerate(np.arange(0.01, 0.02, 0.01)):
def simuLC(x1,color,nproc,input_orig,cad_orig,outdir,zmin,zmax):
    timeRef = time.time()
    #zrange = list(np.arange(0.01, 1.1, 0.01))
    zrange = list(np.arange(zmin, zmax,0.01))
    #zrange = list(np.arange(0.005, 0.01, 0.005))
    #zrange = [0.005]
    
    nz = len(zrange)
    delta = nz
    if nproc > 1:
        delta = int(delta/(nproc))

    batch = range(0,nz,delta)
    if nz not in batch:
        batch = np.append(batch,nz)
    print(batch)

    result_queue = multiprocessing.Queue()

    for i in range(len(batch)-1):
        ida = batch[i]
        idb = batch[i+1]
        p=multiprocessing.Process(name='Subprocess-'+str(i),target=zLoop,args=(input_orig, cad_orig, zrange[ida:idb], x1, color, outdir, 200+ida,i,result_queue))
        p.start()
        print('start',i)

        print('end simu',time.time()-timeRef)

def lcDeriv(x1,color, outdir_final,outdir):
    
    lc_out = outdir_final+'/LC_'+str(x1)+'_'+str(color)+'.hdf5'
    if os.path.exists(lc_out):
        os.remove(lc_out)
    
    files =glob.glob(outdir+'/'+str(x1)+'_'+str(color)+'/*')
    
    for ilc,fi in enumerate(files):
            print('hello',fi)
            Process_LC(fi,lc_out,200+ilc,x1,color)

    # now stack the file produced
    tab_tot = Table()

    
    fi = h5py.File(lc_out, 'r')
    keys = fi.keys()

    for kk in keys:
        
        tab_b = Table.read(fi, path=kk)
        
        if tab_b is not None:
            tab_tot = vstack([tab_tot, tab_b],metadata_conflicts='silent')

    newFile = lc_out.replace('.hdf5','_vstack.hdf5')
    r = tab_tot.to_pandas().values.tolist()
    tab_tot.to_pandas().to_hdf(newFile, key='s')


parser = OptionParser()
parser.add_option("--x1", type="float", default=0.0, help="filter [%default]")
parser.add_option("--color", type="float", default=0.0, help="filter [%default]")
parser.add_option("--nproc", type="int", default=1, help="filter [%default]")
parser.add_option("--simul", type="int", default=1, help="perform simulation [%default]")
parser.add_option("--lcdiff", type="int", default=0, help="produce LC with flux derivatives[%default]")
parser.add_option("--zmin", type="float", default=0.01, help="zmin [%default]")
parser.add_option("--zmax", type="float", default=1.0, help="zmax [%default]")
opts, args = parser.parse_args()

#Example on how to use this code:
# to generate LCs:
#python run_scripts/templates/run_simulation_template.py --simul 1 --lcdiff 0 --x1 -2.0 --color 0.2 --nproc 8
# to generate LCs with derivatives:
# python run_scripts/templates/run_simulation_template.py --simul 0 --lcdiff 1 --x1 0.0 --color 0.0

input_orig = 'input/templates/param_fakesimulation_template.yaml'
cad_orig = 'input/templates/Fake_cadence_template.yaml'
outDir = '/sps/lsst/data/dev/pgris/Templates_new'
outDirFinal = '/sps/lsst/data/dev/pgris/Templates_final_new'
if not os.path.isdir(outDir):
    os.makedirs(outDir)
if not os.path.isdir(outDirFinal):
    os.makedirs(outDirFinal)


x1 = opts.x1
color = opts.color
nproc = opts.nproc
simul = opts.simul
lcdiff = opts.lcdiff
zmin = opts.zmin
zmax = opts.zmax



if simul:
    simuLC(x1,color,nproc,input_orig,cad_orig,outDir,zmin,zmax)

if lcdiff:
    lcDeriv(x1,color,outDirFinal,outDir)
