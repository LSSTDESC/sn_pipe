import os
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--figDir", type="str", default='figures',
                  help="fig directory [%default]")
parser.add_option("--movieDir", type="str", default='movies',
                  help="directory where movies will be placed[%default]")
parser.add_option("--prefix", type="str", default='healpix36503',
                  help="prefix for the figs to assemble as movie [%default]")

opts, args = parser.parse_args()

figDir = opts.figDir
movDir = opts.movieDir
prefix = opts.prefix

cmd = 'ffmpeg -v verbose -r 2 -s 1920x1080 -f image2 -i {}/{}_%00d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p {}/{}.mp4 -y'.format(
    figDir, prefix, movDir, prefix)

print(cmd)


os.system(cmd)
