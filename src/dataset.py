"""
This is the module of the dataset,including table loading, attribute analysis.
"""

import os
import pandas as pd
import numpy as np
from .constants import DATA_ROOT,TEMP_ROOT
from collections import OrderedDict
class Table(object):
    """
    This is a table class, read the data and create the table, parse the table attribute information.
    """
    def __init__(self, dataset):
        """
        Read the data and initialize the table to get the number of rows and columns of the table,
        as well as information about each attribute

        Args:
            dataset:The data set to read
        """
        self.dataset = dataset
        self.name = f"{self.dataset}"
        # load data
        self.data = pd.read_pickle(TEMP_ROOT / f"{self.name}.pkl")
        self.data_size_mb = self.data.values.nbytes / 1024 / 1024
        self.row_num = self.data.shape[0]
        self.col_num = len(self.data.columns)

        # parse columns
        self.parse_columns()
       
    def parse_columns(self):
        self.columns = OrderedDict([(col, Column(col, self.data[col])) for col in self.data.columns])

    def __repr__(self):
        return f"Table {self.name} ({self.row_num} rows, {self.data_size_mb:.2f}MB, columns:\n{os.linesep.join([repr(c) for c in self.columns.values()])})"
class Column(object):
    """
    This is a column class that reads a column of table  and parses it to get some information about the column.  
    """
    def __init__(self, name, data):
        """
        Initializes information about the column, including type, length, maximum and minimum values.  

        Args:
            name: Name of attribute
            data: Columns of data
        """
        self.name = name
        self.dtype = data.dtype

        # parse vocabulary
        self.vocab, self.has_nan = self.__parse_vocab(data)
        self.vocab_size = len(self.vocab)
        self.minval = self.vocab[1] if self.has_nan else self.vocab[0]
        self.maxval = self.vocab[-1]

    def __repr__(self):
        """
        Report column information.
        """
        return f'Column({self.name}, type={self.dtype}, vocab size={self.vocab_size}, min={self.minval}, max={self.maxval}, has NaN={self.has_nan})'

    def __parse_vocab(self, data):
        # pd.isnull returns true for both np.nan and np.datetime64('NaT').
        is_nan = pd.isnull(data)
        contains_nan = np.any(is_nan)
        # NOTE: np.sort puts NaT values at beginning, and NaN values at end.
        # For our purposes we always add any null value to the beginning.
        vs = np.sort(np.unique(data[~is_nan]))
        if contains_nan:
            vs = np.insert(vs, 0, np.nan)
        return vs, contains_nan


def load_table(dataset: str) -> Table:
    table = Table(dataset)
    return table


def csv_to_pkl(table_name: str):
    """
    This function converts CSV file to pickle file.

    Args:
        table_name:The name or path of the csv table
    """
    table_path = DATA_ROOT / f"{table_name}.csv"
    temp_table_path = TEMP_ROOT 
    if not temp_table_path.exists():
        temp_table_path.mkdir()
    pkl_path = temp_table_path / f"{table_name}.pkl"
    if pkl_path.is_file():
        return
    df = pd.read_csv(table_path)
    df.to_pickle(pkl_path)


