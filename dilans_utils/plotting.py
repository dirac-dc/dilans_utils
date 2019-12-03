import sys
from pathlib import Path

import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from dateutil import parser


def remove_timezone(date):
    return parser.parse(re.sub("T.+","",str(date))).replace(tzinfo=None)


def prepfg(x, y, figsize = None):
    if not figsize:
        figsize = (20,10*x)
        
    fig, axes = plt.subplots(x, y, figsize=figsize)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    return fig, axes


def label(ax, xlabel, ylabel, title,  x_fs=20, y_fs=20, t_fs=30):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax

def plot_residuals(residuals, axes=None):
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

