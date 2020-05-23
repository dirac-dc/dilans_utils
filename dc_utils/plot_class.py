import sys
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from dateutil import parser
from colour import Color

mpl.rcParams['figure.dpi']= 300
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlotClass:
    
    # TODO: see if it is worth using the below
    TITLE_SIZE = 30
    TITLE_TO_LABEL_RATIOS = {"axis_label": 0.66666, "axis_tick_label": 0.33333}


    def __init__(self):
        self.fig = None

    def gen_shades(self, length, col="green"):
        
        primary = {"red": [1,0,0], "green": [0,1,0], "blue": [0,0,1]}
        
        if col not in primary.keys():
            raise ValueError("Only red, green or blue allowed")

        rgb = primary[col]
        intervals = np.linspace(0, 1, length)
        for i in range(length):
            yield Color(rgb=[intervals[i] if x != 1 else x for x in rgb])

    def remove_timezone(self, date):
        """Cleans up the presentation of dates for axis labels when the dates are timestamps"""
        return parser.parse(re.sub("T.+","",str(date))).replace(tzinfo=None)

    def prepfg(self, x, y, figsize = None, dpi=72, flatten=True):
        """The same as ple.subplot but with more intelligent sizing and stores the 
        'fig' created for use by 'self.format'"""

        if not figsize:
            figsize = (15,5*x)
            
        self.fig, axes = plt.subplots(x, y, figsize=figsize, dpi=dpi)
        logger.warning(f"'fig' attribute has been set: {self.fig}")
         
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        if flatten:  
            return self.fig, np.ravel(axes)
        else:
            return self.fig, axes

    def format_ax(self, ax, xlabel=None, ylabel=None, title=None, fig=None, xt_fs=10, yt_fs=10, x_fs=20, y_fs=20, t_fs=30, leg_fs="large"):
        """Aims to format the fontsizes of labels as matplotlibs defaults are terrible"""

        if self.fig or fig:
            fig = fig if fig else self.fig
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            if (15 < width < 16) and (7 < height < 8):
                xt_fs, yt_fs, x_fs, y_fs, t_fs = 20, 20, 30, 30, 40
                leg_fs = "xx-large"

        ax.tick_params(axis='x', which='major', labelsize=xt_fs)
        ax.tick_params(axis='x', which='minor', labelsize=xt_fs)
        ax.tick_params(axis='y', which='major', labelsize=yt_fs)
        ax.tick_params(axis='y', which='minor', labelsize=yt_fs)

        ax.set_xlabel(xlabel, fontsize=x_fs) if xlabel else ax.xaxis.label.set_size(x_fs)
        ax.set_ylabel(ylabel, fontsize=y_fs) if ylabel else ax.yaxis.label.set_size(y_fs)
        ax.set_title(title, fontsize=t_fs) if title else ""
        ax.legend(fontsize=leg_fs)

        return ax

    def format_fig(self, fig):
        """Applies 'self.format_ax' to all axes of a figure"""
        for ax in fig.axes:
            self.format_ax(ax)

    def plot_residuals(self, residuals, axes=None):
        if not axes:
            fig, axes = prepfg(1,2)
        
        # create residuals timeseries
        _ = label(axes[0], "Month", "Error", "Residuals Plot")
        
        outliers = {}
        
        # normal dbn outliers
        cutoff = residuals.std()*1.96
        axes[0].axhline(y=cutoff, color="r", ls="dashed", label="Normality Assumptions")
        axes[0].axhline(y=-cutoff, color="r", ls="dashed", label="")
        n_mask = ((residuals > cutoff) | (residuals < -cutoff))
        outliers["normal_outliers"] = res[n_mask]
        axes[0].scatter(residuals[n_mask].index, residuals[n_mask].values,  color="red", label="Normal Dbn Outliers",  alpha=0.5, s=100)

        # twitter anomaly outiers
        out = tad.anomaly_detect_vec(residuals.dropna(), period=12, max_anoms=0.02, direction="pos", plot=True)
        axes[0].scatter(out.index, out.values, color="green", label="Twitter Anomaly Detector",  alpha=0.5, s=200)
        axes[0].legend(fontsize=20)
        outliers["twitter_anomaly_detection"] = out
        
        # create residual histogram
        _ = residuals.hist(ax=axes[1])
        _ = label(axes[1], "Error", "Density", "Residuals Hist Plot")
        
        return outliers, axes

plotter = PlotClass()
