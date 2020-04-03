from pathlib import Path

import pandas as pd


class Io:

    DF_STYLESHEET_LINK = "https://cdn.jupyter.org/notebook/5.1.0/style/style.min.css"

    def df_html(df: pd.DataFrame, filepath: Path):
        """Write a data frame to an html script with a linked stylesheet.
        TODO: The stylesheet needs to be improved along with tag insertions.
        """

        stylesheet_str = """<link rel="stylesheet" href="{self.DF_STYLESHEET_LINK}">\n"""
        with open(filepath, "w") as f:
            f.writelines(stylesheet_str + df.to_html())
