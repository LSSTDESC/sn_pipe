import os
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--figDir", type="str", default='figures',
                  help="fig directory [%default]")
parser.add_option("--movieDir", type="str", default='movies',
                  help="directory where movies will be placed[%default]")
parser.add_option("--prefix", type="str", default='healpix36503',
                  help="prefix for the figs to assemble as movie [%default]")
parser.add_option("--extens", type="str", default='jpg',
                  help="extension of the figures to assemble as a movie [%default]")
parser.add_option("--rate", type=int, default=10,
                  help="movie rate [%default]")
opts, args = parser.parse_args()

figDir = opts.figDir
movDir = opts.movieDir
prefix = opts.prefix
extens = opts.extens
rate = opts.rate

if not os.path.exists(movDir):
    os.mkdir(movDir)

cmd = 'ffmpeg -v verbose -r {} -s 1920x1080 -f image2 -i {}/{}_%00d.{} -vcodec libx264 -crf 25  -pix_fmt yuv420p {}/{}.mp4 -y'.format(
    rate, figDir, prefix, extens, movDir, prefix)

print(cmd)


os.system(cmd)
