import sys
from typing import Union

import googlemaps
import numpy as np
import pandas as pd
import re
from datetime import datetime
import matplotlib.colors as mcolors
from dateutil import parser


class CleanClass:
    POSSIBLE_COLOURS = list(mcolors.CSS4_COLORS.keys())
    DISTINCT_COLOURS = ["green", "blue", "orange", "black", "yellow", "grey","red"]

    @staticmethod
    def coalesce(dframe: pd.DataFrame, prefix: str, name=None) -> pd.DataFrame:
        '''To be used after a pandas merge to coalesce prefix_x, prefix_y columns'''
        df = dframe.copy()
        coi = [col for col in df.columns if prefix in col]
        merged = df[coi[0]]
        for col in coi[1:]:
            merged = merged.combine_first(df[col])
        if not name:
            name = prefix
        df[name] = merged
        return df.drop(columns=coi)

    def find_date_cols(self, df):
        date_cols = []
        for col, values in df.iteritems():
            try:
                test = pd.to_datetime(values.dropna().sample(1).values).year[0]
                if test > 2000:
                    date_cols.append(col)
            except:
                pass
        return date_cols


    def remove_timezone(self, date):
        return parser.parse(re.sub("T.+","",str(date))).replace(tzinfo=None) if pd.notnull(date) else np.nan


    def convert_dates(self, df, date_cols=None, spec_format={}, timezone=False, verbose=False):
        
        df = df.copy()

        if spec_format:
            for dcol, dformat in spec_format.items():
                df[dcol] = df[dcol].apply(lambda x: pd.to_datetime(x, format=dformat))

        if not date_cols:
            date_cols = self.find_date_cols(df.drop(columns = list(spec_format.keys())))
        
        df[date_cols] = df[date_cols].apply(lambda x: pd.to_datetime(x), axis=0)

        if not timezone:
            df[date_cols] = df[date_cols].apply(lambda col: col.apply(self.remove_timezone))

        if verbose:
            for date_col in date_cols:
                temp = df[date_col].dropna()
                print("{} : \n    earliest {}; latest {}".format(date_col, temp.min(), temp.max()))

        return df, date_cols


    def floatHourToTime(self, fh):
        h, r = divmod(fh, 1)
        m, r = divmod(r*60, 1)
        return (
            int(h),
            int(m),
            int(r*60),
        )


    def excel_date_conversion(self, excel_date):
        if pd.isnull(excel_date):
            return excel_date
        else:
            dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + int(excel_date) - 2)
            hour, minute, second = self.floatHourToTime(excel_date % 1)
            dt = dt.replace(hour=hour, minute=minute, second=second)
        return dt


    def convert_excel_dates(self, df, date_cols):
        df[date_cols] = (df[date_cols].apply(lambda x: x.apply(excel_date_conversion),axis=0))
        for date_col in date_cols:
            temp = df[date_col].dropna()
#         print("{} : \n    earliest {}; latest {}".format(date_col, temp.min(), temp.max()))
        return df

    def standardize_string(self, x: str) -> str:
        """For use with only column headings or index like attributes, not for general cleaning.
        Allows for clearer and consistent reading replacing spaces with underscores.
        """
        if pd.isnull(x):
            return np.nan
        remove_characters = "(%|\?|\.|\(.*?\))"
        replace_characters = "(\s|/|\-)"
        x = re.sub(remove_characters, "", x)
        x = re.sub(replace_characters, "_", x).lower()
        x = x.strip("_ ")
        x = re.sub("\_+", "_", x)
        return x
        

    def standardize_names(self, df: pd.DataFrame) -> pd.DataFrame: 
        """Standardizes column headings"""
        df.columns = [self.standardize_string(x) for x in df.columns]
        return df

    def get_named_colours(n):
        return col_names[n%(len(POSSIBLE_COLOURS))]


    def clean_excel(self, path, header_row=0, sheet_name=0):
        # TODO find the header row automatically
        data = pd.read_excel(path, header = header_row, sheet_name=sheet_name)
        data = self.standardize_names(data)
        if np.nan in data.columns.to_list():
            data = data.drop(columns=[np.nan])
        data = data.dropna(axis=1, how="all")
        return self.convert_dates(data)

clean = CleanClass()
