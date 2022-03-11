"""
This is a module for generating queries.
Each query is generated through three steps:
    1.Choose a set of attributes to place predicates.
    2.Select the query center for each predicate.
    3.Determine the operator for each predicate (as well as widths for range predicates).
"""

import random
import numpy as np
import pandas as pd
from typing_extensions import Protocol
from typing import Dict, Optional, Tuple, List, Any
from .dataset import Table,Column
from .query import Query,new_query
from .dtpye import is_categorical
"""====== Attribute Selection Functions ======"""
#table.data.columns
class AttributeSelFunc(Protocol):
    def __call__(self, table: Table, params: Dict[str, Any]) -> List[str]: ...
def asf_pred_number(table: Table, params: Dict[str, Any]) -> List[str]:
    """
    Select attribute randomly.
    Args:
        table:The table for attribute  selection
        params:
            whitelist:Select attributes from the whitelist
            blacklist:Select attributes that are not on the blacklist
            nums:Maximum number of attributes to select
    Retrun: A list of selected attributes
    """
    if 'whitelist' in params:
        attr_domain = params['whitelist']
    else:
        blacklist = params.get('blacklist') or []
        attr_domain = [c for c in list(table.data.columns) if c not in blacklist]
    nums = params.get('nums')
    nums = nums or range(1, len(attr_domain)+1)
    num_pred = np.random.choice(nums)
    assert num_pred <= len(attr_domain)
    return np.random.choice(attr_domain, size=num_pred, replace=False)

def asf_comb(table: Table, params: Dict[str, Any]) -> List[str]:
    """
    Select attributes based on input.
    Args:
         table:The table for attribute  selection
         params:
            comb:attribute to select
    Return: A list of selected attributes.
    """
    assert 'comb' in params and type(params['comb']) == list, params
    for c in params['comb']:
        assert c in table.columns, c
    return params['comb']

"""====== Center Selection Functions ======"""
class CenterSelFunc(Protocol):
    def __call__(self, table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]: ...
ROW_CACHE = None
GLOBAL_COUNTER = 1000

#Optional params:data_from:Center selection starts here
def csf_distribution(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    """
    Randomly select 1000 rows as candidate for the data center.
    Args:
        table:The table for attribute  selection
        attrs:A list of selected attributes.
        params:
            data_from:Start from data_from to select the rows
    Return:A list of data for selected attributes.
    """
    global GLOBAL_COUNTER
    global ROW_CACHE
    if GLOBAL_COUNTER >= 1000:
        data_from = params.get('data_from') or 0
        ROW_CACHE = np.random.choice(range(data_from, len(table.data)), size=1000)
        GLOBAL_COUNTER = 0
    row_id = ROW_CACHE[GLOBAL_COUNTER]
    GLOBAL_COUNTER += 1
    #  data_from = params.get('data_from') or 0
    #  row_id = np.random.choice(range(data_from, len(table.data)))
    return [table.data.at[row_id, a] for a in attrs]

def csf_vocab_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    """
    #Randomly select a center for each attribute
    """
    centers = []
    for a in attrs:
        col = table.columns[a]
        centers.append(np.random.choice(col.vocab))
    return centers


#def csf_domain_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    centers = []
    for a in attrs:
        col = table.columns[a]
        if is_categorical(col.dtype): # randomly pick one point from domain for categorical
            centers.append(np.random.choice(col.vocab))
        else: # uniformly pick one point from domain for numerical
            centers.append(random.uniform(col.minval, col.maxval))
    return centers


"""====== Width Selection Functions ======"""
class WidthSelFunc(Protocol):
    def __call__(self, table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query: ...

def parse_range(col: Column, left: Any, right: Any) -> Optional[Tuple[str, Any]]:
    """
    Select predicates based on range
    """
    if left <= col.minval:
        return ('<=', right)
    if right >= col.maxval:
        return ('>=', left)
    return ('[]', (left, right))

def wsf_uniform(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    """
    Uniformly select a width between the maximum and the minimum.
    """
    query = new_query(table, ncols=len(attrs))
    for a, c in zip(attrs, centers):
        # NaN/NaT literal can only be assigned to = operator
        if pd.isnull(c) or is_categorical(table.columns[a].dtype):
            query.predicates[a] = ('=', c)
            continue
        col = table.columns[a]
        width = random.uniform(0, col.maxval-col.minval)
        query.predicates[a] = parse_range(col, c-width/2, c+width/2)
    return query

def wsf_exponential(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    """
    Select width by exponential distribution
    """
    query = new_query(table, ncols=len(attrs))
    for a, c in zip(attrs, centers):
        # NaN/NaT literal can only be assigned to = operator
        if pd.isnull(c) or is_categorical(table.columns[a].dtype) :
            query.predicates[a] = ('=', c)
            continue
        col = table.columns[a]
        lmd = 1 / ((col.maxval - col.minval) / 10)
        width = random.expovariate(lmd)
        query.predicates[a] = parse_range(col, c-width/2, c+width/2)
    return query
#naru
def wsf_naru(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    """
    1.Select an operator from ">=,<=,=" at random.
    2.If the column size >=10, combine the data center and operator into predicates
    3.Else, "="
    """
    query = new_query(table, ncols=len(attrs))
    ops = np.random.choice(['>=', '<=', '='], size=len(attrs))
    for a, c, o in zip(attrs, centers, ops):
        if table.columns[a].vocab_size >= 10:
            query.predicates[a] = (o, c)
        else:
            query.predicates[a] = ('=', c)
    return query

def wsf_equal(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    """
    Only equality predicates are generated
    """
    query = new_query(table, ncols=len(attrs))
    for a, c in zip(attrs, centers):
        query.predicates[a] = ('=', c)
    return query

class QueryGenerator(object):
    table: Table
    attr: Dict[AttributeSelFunc, float]
    center: Dict[CenterSelFunc, float]
    width: Dict[WidthSelFunc, float]
    attr_params: Dict[str, Any]
    center_params: Dict[str, Any]
    width_params: Dict[str, Any]

    def __init__(
            self, table: Table,
            attr: Dict[AttributeSelFunc, float],#(func,p)
            center: Dict[CenterSelFunc, float],
            width: Dict[WidthSelFunc, float],
            attr_params: Dict[str, Any],
            center_params: Dict[str, Any],
            width_params: Dict[str, Any]
            ) -> None:
        self.table = table
        self.attr = attr
        self.center = center
        self.width = width
        self.attr_params = attr_params
        self.center_params = center_params
        self.width_params = width_params

    def generate(self) -> Query:
        """
        Generate query.
        """
        attr_func = np.random.choice(list(self.attr.keys()), p=list(self.attr.values()))
        attr_lst = attr_func(self.table, self.attr_params)
        center_func = np.random.choice(list(self.center.keys()), p=list(self.center.values()))
        center_lst = center_func(self.table, attr_lst, self.center_params)
        width_func = np.random.choice(list(self.width.keys()), p=list(self.width.values()))
        return width_func(self.table, attr_lst, center_lst, self.width_params)
