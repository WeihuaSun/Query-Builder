import random
import numpy as np
import pandas as pd
from typing_extensions import Protocol
from typing import Dict, NamedTuple, Optional, Tuple, List, Any
from dataset import Table,Column
from workload import Query,new_query
"""====== Attribute Selection Functions ======"""
#table.data.columns
class AttributeSelFunc(Protocol):
    def __call__(self, table: Table, params: Dict[str, Any]) -> List[str]: ...
#给定属性数量，属性白名单和黑名单，进行属性选择
def asf_pred_number(table: Table, params: Dict[str, Any]) -> List[str]:
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
#给定属性，判断属性是否在里面
def asf_comb(table: Table, params: Dict[str, Any]) -> List[str]:
    assert 'comb' in params and type(params['comb']) == list, params
    for c in params['comb']:
        assert c in table.columns, c
    return params['comb']
#naru的办法
def asf_naru(table: Table, params: Dict[str, Any]) -> List[str]:
    num_filters = np.random.randint(5, 12)
    return np.random.choice(table.data.columns, size=num_filters, replace=False)
"""====== Center Selection Functions ======"""
class CenterSelFunc(Protocol):
    def __call__(self, table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]: ...
DOMAIN_CACHE = {}
# This domain version makes sure that query's cardinality > 0
def csf_domain(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    global DOMAIN_CACHE
    key = tuple(sorted(attrs))
    if key not in DOMAIN_CACHE:
        data_from = params.get('data_from') or 0
        DOMAIN_CACHE[key] = table.data[data_from:][attrs].drop_duplicates().index
        assert len(DOMAIN_CACHE[key]) > 0, key
    #  L.debug(f'Cache size: {len(DOMAIN_CACHE)}')
    row_id = np.random.choice(DOMAIN_CACHE[key])
    return [table.data.at[row_id, a] for a in attrs]

ROW_CACHE = None
GLOBAL_COUNTER = 1000
def csf_distribution(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
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

def csf_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    row_ids = np.random.choice(len(table.data), len(attrs))
    return [table.data.at[i, a] for i, a in zip(row_ids, attrs)]

def csf_vocab_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    centers = []
    for a in attrs:
        col = table.columns[a]
        centers.append(np.random.choice(col.vocab))
    return centers

def csf_domain_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    centers = []
    for a in attrs:
        col = table.columns[a]
        if is_categorical(col.dtype): # randomly pick one point from domain for categorical
            centers.append(np.random.choice(col.vocab))
        else: # uniformly pick one point from domain for numerical
            centers.append(random.uniform(col.minval, col.maxval))
    return centers

def csf_naru(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    row_id = np.random.randint(0, len(table.data))
    return [table.data.at[row_id, a] for a in attrs]

def csf_naru_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    row_ids = np.random.choice(len(table.data), len(attrs))
    return [table.data.at[i, a] for i, a in zip(row_ids, attrs)]

"""====== Width Selection Functions ======"""
class WidthSelFunc(Protocol):
    def __call__(self, table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query: ...
#给定左值和右值，选择对应的谓词
def parse_range(col: Column, left: Any, right: Any) -> Optional[Tuple[str, Any]]:
    if left <= col.minval:
        return ('<=', right)
    if right >= col.maxval:
        return ('>=', left)
    return ('[]', (left, right))
#在最大值和最小值之间均匀选择宽度
def wsf_uniform(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
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
#指数分布
def wsf_exponential(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    query = new_query(table, ncols=len(attrs))
    for a, c in zip(attrs, centers):
        # NaN/NaT literal can only be assigned to = operator
        if pd.isnull(c) or is_categorical(table.columns[a].dtype):
            query.predicates[a] = ('=', c)
            continue
        col = table.columns[a]
        lmd = 1 / ((col.maxval - col.minval) / 10)
        width = random.expovariate(lmd)
        query.predicates[a] = parse_range(col, c-width/2, c+width/2)
    return query
#naru文章提出的方法1.选择谓词2.超过10个用选择的谓词，否则用相等谓词
def wsf_naru(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    query = new_query(table, ncols=len(attrs))
    ops = np.random.choice(['>=', '<=', '='], size=len(attrs))
    for a, c, o in zip(attrs, centers, ops):
        if table.columns[a].vocab_size >= 10:
            query.predicates[a] = (o, c)
        else:
            query.predicates[a] = ('=', c)
    return query
#只取相等谓词，等于中间值
def wsf_equal(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
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
            attr: Dict[AttributeSelFunc, float],
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
        attr_func = np.random.choice(list(self.attr.keys()), p=list(self.attr.values()))
      
        attr_lst = attr_func(self.table, self.attr_params)

        center_func = np.random.choice(list(self.center.keys()), p=list(self.center.values()))
        
        center_lst = center_func(self.table, attr_lst, self.center_params)

        width_func = np.random.choice(list(self.width.keys()), p=list(self.width.values()))
       
        return width_func(self.table, attr_lst, center_lst, self.width_params)




