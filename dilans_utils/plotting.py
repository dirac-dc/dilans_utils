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


def label(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax

