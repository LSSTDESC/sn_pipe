#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:56:04 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import ffmpeg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as manimation
import numpy as np
import pandas as pd


class Video:
    def __init__(self, data, title='test', fps=24):

        self.data = data

        # prepare frame
        self.fig = plt.figure()
        self.ax = self.fig.gca()

        writer_type = 'ffmpeg'
        extension = 'mp4'

        Writer = manimation.writers[writer_type]
        # Writer = manimation.writers[writer_type]
        metadata = dict(title=title, artist='Matplotlib',
                        comment=title)
        # writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=6000)
        writer = Writer(fps=fps, metadata=metadata, bitrate=6000)
        # writer = anim.FFMpegWriter(fps=30, codec='hevc')
        Name_mp4 = '{}.{}'.format(title, extension)
        print('name for saving', Name_mp4)
        with writer.saving(self.fig, Name_mp4, 250):
            self.loopdata(writer=writer)

    def loopdata(self, writer):

        nv = len(self.data)
        for i in range(1, nv):
            dd = self.data[:i]
            self.ax.plot(dd['x'], dd['y'], 'ko')
            self.fig.canvas.flush_events()
            if writer:
                writer.grab_frame()
            self.ax.clear()


x = np.arange(1, 100, 2)
data = pd.DataFrame(x, columns=['x'])
data['y'] = 3*data['x']-7


myvideo = Video(data, fps=2)
