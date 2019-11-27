import sys

import googlemaps
import numpy as np
import pandas as pd
import re
from datetime import datetime
import matplotlib.colors as mcolors
from dateutil import parser

POSSIBLE_COLOURS = list(mcolors.CSS4_COLORS.keys())
DISTINCT_COLOURS = ["green", "blue", "orange", "black", "yellow", "grey","red"]


def find_date_cols(df):
    date_cols = []
    for col, values in df.iteritems():
        try:
            test = pd.to_datetime(values.dropna().sample(1).values).year[0]
            if test > 2000:
                date_cols.append(col)
        except:
            pass
    return date_cols


def remove_timezone(date):
    return parser.parse(re.sub("T.+","",str(date))).replace(tzinfo=None) if pd.notnull(date) else np.nan


def convert_dates(df, date_cols=None, timezone=False, verbose=False):
    if not date_cols:
        date_cols = find_date_cols(df)
    
    df[date_cols] = df[date_cols].apply(lambda x: pd.to_datetime(x), axis=0)

    if not timezone:
        df[date_cols] = df[date_cols].apply(lambda col: col.apply(remove_timezone))

    if verbose:
        for date_col in date_cols:
            temp = df[date_col].dropna()
            print("{} : \n    earliest {}; latest {}".format(date_col, temp.min(), temp.max()))

    return df, date_cols


def floatHourToTime(fh):
    h, r = divmod(fh, 1)
    m, r = divmod(r*60, 1)
    return (
        int(h),
        int(m),
        int(r*60),
    )


def excel_date_conversion(excel_date):
    if pd.isnull(excel_date):
        return excel_date
    else:
        dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + int(excel_date) - 2)
        hour, minute, second = floatHourToTime(excel_date % 1)
        dt = dt.replace(hour=hour, minute=minute, second=second)
    return dt


def convert_excel_dates(df, date_cols):
    df[date_cols] = (df[date_cols].apply(lambda x: x.apply(excel_date_conversion),axis=0))
    for date_col in date_cols:
        temp = df[date_col].dropna()
#         print("{} : \n    earliest {}; latest {}".format(date_col, temp.min(), temp.max()))
    return df


def standardize_names(df):
    remove_characters = "(%|\?|\.|\(.*?\))"
    replace_characters = "(\s|/|\-)"
    temp = df.columns
    temp = [re.sub(remove_characters, "", x) for x in temp]
    temp = [re.sub(replace_characters, "_", x).lower() for x in temp]
    temp = [x.strip("_ ") for x in temp]
    df.columns = temp
    return df
    

def get_named_colours(n):
    return col_names[n%(len(POSSIBLE_COLOURS))]


def clean_excel(path, header_row=0, sheet_name=0):
    data = pd.read_excel(path, header = header_row, sheet_name=sheet_name)
    data = standardize_names(data)
    if np.nan in data.columns.to_list():
        data = data.drop(columns=[np.nan])
    data = data.dropna(axis=1, how="all")
    return convert_dates(data)
