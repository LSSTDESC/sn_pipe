### Usage: run_sn_fit.py [options] ###
<pre>

Options:
  -h, --help           show this help message and exit
  --dirFiles=DIRFILES  location dir of the
                       files[/sps/lsst/users/gris/Output_Simu_pipeline_0]
  --prodid=PRODID      db name [Test]
  --outDir=OUTDIR      output dir [/sps/lsst/users/gris/Output_Fit_0]
  --nproc=NPROC        number of proc [1]
  --mbcov=MBCOV        mbcol calc [0]
  --display=DISPLAY    to display fit in real-time[0]
  --fitter=FITTER      fitter to use [sn_cosmo]


</pre>

## Example
<ul>
<li> Let us suppose that some simulations have been run using the sn_simulation package. In the output simulation directory, three files should be available: prodid.yaml, Simu_prodid_num.hdf5, LC_prodid_num.hdf5, where prodid has been defined in the yaml file used for the simulation. num is an integer corresponding to the multiprocessing configuration used at the simulation stage.

ls Output_Simu_380.0_800.0_ebvofMW_0.1
sn_cosmo_Fake_Fake_DESC_seas_-1_0.0_0.0_380.0_800.0_ebvofMW_0.1.yaml 
Simu_sn_cosmo_Fake_Fake_DESC_seas_-1_0.0_0.0_380.0_800.0_ebvofMW_0.1_0.hdf5 
LC_sn_cosmo_Fake_Fake_DESC_seas_-1_0.0_0.0_380.0_800.0_ebvofMW_0.1_0.hdf5

<li> In that case, the fit of the light curves will be made using:
     <ul>
     <li> python run_scripts/fit_sn/run_sn_fit.py --dirFiles Output_Simu_380.0_800.0_ebvofMW_0.1 --prodid sn_cosmo_Fake_Fake_DESC_seas_-1_0.0_0.0_380.0_800.0_ebvofMW_0.1_0 --nproc 4 --outDir Output_Fit
     </ul>

<li> Output directory after processing:
<ul>
<li> ls Output_Fit

Fit_sn_cosmo_Fake_Fake_DESC_seas_-1_0.0_0.0_380.0_800.0_ebvofMW_0.1_0_sn_cosmo.hdf5
</ul>
</ul>