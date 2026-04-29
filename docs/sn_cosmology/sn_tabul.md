# $${\color{red} Distance \space modulus \space tabulated \space values}$$

## Usage: run_scripts/cosmology/run_cosmo_tabul.py [options]

<pre>

Script to estimate distmod in 3D

Options:
  -h, --help            show this help message and exit
  --deparams=DEPARAMS   DE eos parameters [w1,w2]
  --cosmofitparams=COSMOFITPARAMS
                        parameters used to estimate distmod [w1,w2,Om0]
  --cosmofitparams_min=COSMOFITPARAMS_MIN
                        fit parameter min values [-1.,-10.,0.2]
  --cosmofitparams_max=COSMOFITPARAMS_MAX
                        fit parameter max values [0.,-1.,0.4]
  --cosmofitparams_delta=COSMOFITPARAMS_DELTA
                        fit parameter n values [0.1,0.1,0.05]
  --declass=DECLASS     DE class to use (w0waCDM/DDE_FLRW) [DDE_FLRW]
  --classloc=CLASSLOC   DE class location
                        (astropy.cosmology/sn_tools.sn_cosmo_model)
                        [sn_tools.sn_cosmo_model]
  --demodel=DEMODEL     DE eos model name [oscilla]
  --deeos=DEEOS         DE eos model [-1.+(w1*z*np.sin(w2*z))/(1+z**2)]
  --H0=H0               DE eos model [70.0]
  --Om0=OM0             Omega_matter [0.3]
  --Ode0=ODE0           Omega_DE [0.7]
  --outName=OUTNAME     prefix for output name [distmod_tabul]
  --outDir=OUTDIR       output directory [../distmod_tabul]

</pre>