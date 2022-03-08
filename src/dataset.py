import os
import pandas as pd
import numpy as np
import pickle
from .constants import DATA_ROOT,PKL_PROTO,TEMP_ROOT
from collections import OrderedDict
class Table(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.name = f"{self.dataset}"
      
        # load data
        self.data = pd.read_pickle(DATA_ROOT / self.dataset / f".pkl")
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
    def __init__(self, name, data):
        self.name = name
        self.dtype = data.dtype

        # parse vocabulary
        self.vocab, self.has_nan = self.__parse_vocab(data)
        self.vocab_size = len(self.vocab)
        self.minval = self.vocab[1] if self.has_nan else self.vocab[0]
        self.maxval = self.vocab[-1]

    def __repr__(self):
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

    def discretize(self, data):
        """Transforms data values into integers using a Column's vocabulary"""

        # pd.Categorical() does not allow categories be passed in an array
        # containing np.nan.  It makes it a special case to return code -1
        # for NaN values.
        if self.has_nan:
            bin_ids = pd.Categorical(data, categories=self.vocab[1:]).codes
            # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
            # add 1 to everybody
            bin_ids = bin_ids + 1
        else:
            # This column has no nan or nat values.
            bin_ids = pd.Categorical(data, categories=self.vocab).codes

        bin_ids = bin_ids.astype(np.int32, copy=False)
        assert (bin_ids >= 0).all(), (self, data, bin_ids)
        return bin_ids



def load_table(dataset: str) -> Table:
    table_path = DATA_ROOT / dataset / f".table.pkl"
    if  table_path.is_file():
        with open(table_path, 'rb') as f:
            table = pickle.load(f)
        return table
    table = Table(dataset)
    dump_table(table)
    return table

def dump_table(table: Table) -> None:
    with open(DATA_ROOT / table.dataset / f"{table.version}.table.pkl", 'wb') as f:
        pickle.dump(table, f, protocol=PKL_PROTO)
def csv_to_pkl(table_name: str):
    table_path = DATA_ROOT / f"{table_name}.csv"
    temp_table_path = TEMP_ROOT / "table"
    if not temp_table_path.exists():
        temp_table_path.mkdir()
    pkl_path = temp_table_path / f"{table_name}.pkl"
    if pkl_path.is_file():
        return
    df = pd.read_csv(table_path)
    df.to_pickle(pkl_path)