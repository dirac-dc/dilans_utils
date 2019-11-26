import pandas as pd
import folium


def plot_latlon(df):
    """Requires df with cols = ["lat", "lon", "name":Opt]
    """
    df.columns = df.columns.map(str.lower)
    assert all([col in df.columns for col in ["lat", "lon"]]), \
        "'lat', 'lon' columns not found"

    m = folium.Map()

    for ind, row in df.iterrows():
        if 'name' in row.index:
            ind = row["name"]
        folium.Marker([row.lat, row.lon], popup=str(ind)).add_to(m)

    _ = folium.map.FitBounds(m.get_bounds()).add_to(m)
    return m
