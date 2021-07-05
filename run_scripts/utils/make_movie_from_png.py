import os


figDir = 'figures_nsn'
movDir = 'movies'
prefix = 'healpix1'

cmd = 'ffmpeg -r 1 -s 1920x1080 -f image2 -i {}/{}_%000d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p {}/{}.mp4 -y'.format(
    figDir, prefix, movDir, prefix)

print(cmd)


os.system(cmd)
