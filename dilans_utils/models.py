import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from dateutil import parser
from fbprophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
from fbprophet.plot import plot_yearly
import tad

from plotting import *

def fb_prophet(ts: pd.Series, seasonality: str = "multiplicative", plot=True):
    '''
    Creates a model using FB Prohet for a timeseries. Creates predicted plot along with 
    with residual plots.

    Parameters
    ----------
    y: 
        Series with datetime index.
    
    seasonality:
        
    Returns
    -------
    fb_prophet_df, (main_ax, resid_ax, hist_resid_ax)
    '''
    fig = plt.figure(constrained_layout=True, figsize=(20,20))
    gs = GridSpec(2,2, figure=fig)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    
    m = Prophet(seasonality_mode="multiplicative")
    ts = ts.to_frame("y")
    ts["ds"] = ts.index.map(lambda x: remove_timezone(x))
    m.fit(ts)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    _ = m.plot(forecast, ax=ax0)
    res = ts["y"] - forecast.set_index("ds")["yhat"]
    
    _ = res.plot(ax=ax1, label="")
    _ = label(ax1, "Month", "Error", "Residual Plot")
    
    forecast["res"] = res.values
    
    forecast["y"] = np.nan
    forecast.update(ts.reset_index().drop(columns=["create_date"]))
    forecast = forecast.set_index("ds")
    
    # create residual plots
    outliers, (ax1, ax2) = plot_residuals(forecast.res, (ax1,ax2))
    
    # plot outliers on main graph
    colors = ["red", "green"]
    for k, out_type in enumerate(outliers):
        mask = forecast.index.isin(outliers[out_type].index)
        ax0.scatter(outliers[out_type].index,
                    forecast[mask].y.values,
                    color=colors[k],
                    label=out_type,
                    alpha=0.5,
                    s=100+k*100)

    ax0.legend(loc="upper left", fontsize=30)
    return forecast, fig, (ax0, ax1, ax2)
