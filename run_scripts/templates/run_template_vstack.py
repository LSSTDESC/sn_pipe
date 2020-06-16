import h5py
import glob
from optparse import OptionParser
import os
from astropy.table import Table, Column, vstack


class LCStack:
    """
    class to stack a set of LC stored as astropy tables

    Parameters
    --------------
    x1: float
     SN x1
    color: float
     SN color
    lcDir: str
      location dir where LCs are
    outDir: str
      output directory where stacked LC will be located
    """

    def __init__(self, x1, color, lcDir, outDir):

         # create output directory
        if not os.path.isdir(outDir):
            os.makedirs(outDir)

        lc_out = outDir+'/LC_'+str(x1)+'_'+str(color)+'.hdf5'
        if os.path.exists(lc_out):
            os.remove(lc_out)

        files = glob.glob(lcDir+'/'+str(x1)+'_'+str(color)+'/LC*')

        for ilc, fi in enumerate(files):
            self.transform_LC(fi, lc_out, 200+ilc, x1, color)

        # now stack the file produced
        tab_tot = Table()

        fi = h5py.File(lc_out, 'r')
        keys = fi.keys()

        for kk in keys:

            tab_b = Table.read(fi, path=kk)
            if tab_b is not None:
                tab_tot = vstack([tab_tot, tab_b], metadata_conflicts='silent')

        newFile = lc_out.replace('.hdf5', '_vstack.hdf5')
        r = tab_tot.to_pandas().values.tolist()
        tab_tot.to_pandas().to_hdf(newFile, key='s')

    def transform_LC(self, fname, lc_out, ilc, x1, color):
        """
        Method to generate a new LC file from existing ones

        Parameters
        ----------------
        fname: str
          LC file name (hdf5)
        lc_out: str
           output file name
        ilc: int
          index used as a key in lc_out
        x1: float
          SN x1
        color: float
          SN color

        """

        f = h5py.File(fname, 'r')
        keys = list(f.keys())

        prefix_key = '_'.join(keys[0].split('_')[:-1])
        last_keys = []
        keys = [1, 10, 100, 1000]
        vals = ['daymax', 'color', 'x1', 'x0']
        tab = Table.read(f, path='{}_0'.format(prefix_key))
        table_new = Table(tab)

        for ii, kk in enumerate(keys):
            tablea = Table.read(f, path='{}_{}'.format(prefix_key, kk))
            tableb = Table.read(f, path='{}_{}'.format(prefix_key, -kk))
            epsilona = tablea.meta['epsilon_'+vals[ii]]
            epsilonb = tableb.meta['epsilon_'+vals[ii]]
            assert((epsilona == -epsilonb))
            col = (tablea['flux']-tableb['flux'])/(2.*epsilona)
            table_new.add_column(Column(col, name='d'+vals[ii]))
        bands = [b[-1] for b in table_new['band']]
        table_new.remove_column('band')
        table_new.add_column(
            Column(bands, name='band', dtype=h5py.special_dtype(vlen=str)))
        #phases = (table_new['time']-table_new['DayMax'])/(1.+table_new['z'])
        #table_new.add_column(Column(phases, name='phase'))
        table_new['x1'] = x1
        table_new['color'] = color
        table_new['z'] = table_new.meta['z']
        table_new['daymax'] = table_new.meta['daymax']

        table_new.write(lc_out, path='lc_'+str(ilc),
                        append=True, compression=True)


parser = OptionParser()
parser.add_option('--x1', type='float', default=0.0, help='SN x1 [%default]')
parser.add_option('--color', type='float', default=0.0,
                  help='SN color [%default]')
parser.add_option('--lcDir', type='str', default='Test_fakes',
                  help='lc directory[%default]')
parser.add_option('--outDir', type='str',
                  default='Templates_final_new', help='output directory for [%default]')
opts, args = parser.parse_args()

LCStack(opts.x1, opts.color, opts.lcDir, opts.outDir)
