!#/bin/bash

echo "processing data"

cmd="python run_scripts/simulation/run_flux_spectra_sn.py --sed=1 --outDir_display=../plot_flux_spectra"

$cmd

echo "creating movie"

cmd='python run_scripts/utils/make_movie_from_png.py --figDir=../plot_flux_spectra --prefix=flux_spectra --outName=flux_spectra --extens=png'

$cmd

echo "displaying movie"

cmd="vlc movies/flux_spectra.mp4"

$cmd
