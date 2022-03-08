import random
from typing import Dict, NamedTuple, Optional, Tuple, List, Any
from collections import OrderedDict
from dataset import Table
from generator import QueryGenerator
import generator
import numpy as np
from dataset import load_table
import csv
class Query(NamedTuple):
    """predicate of each attritbute are conjunctive"""
    predicates: Dict[str, Optional[Tuple[str, Any]]]
    ncols: int
def new_query(table: Table, ncols) -> Query:
    return Query(predicates=OrderedDict.fromkeys(table.data.columns, None),
                 ncols=ncols)

def generate_workload(
    seed: int, dataset: str, version: str,
    name: str, no_label: bool, old_version: str, win_ratio: str,
    params: Dict[str, Dict[str, Any]]
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    attr_funcs = {getattr(generator, f"asf_{a}"): v for a, v in params['attr'].items()}
    center_funcs = {getattr(generator, f"csf_{c}"): v for c, v in params['center'].items()}
    width_funcs = {getattr(generator, f"wsf_{w}"): v for w, v in params['width'].items()}

    table = load_table(dataset, version)
   
    qgen = QueryGenerator(
            table=table,
            attr=attr_funcs,
            center=center_funcs,
            width=width_funcs,
            attr_params=params.get('attr_params') or {},
            center_params=params.get('center_params') or {},
            width_params=params.get('width_params') or {})

    queryset = {}
    for group, num in params['number'].items():
        queries = []
        for i in range(num):
            queries.append(qgen.generate())
        queryset[group] = queries

    #dump_sqls(dataset, name, queryset)

    if no_label:
        L.info("Finish without generating corresponding ground truth labels")
        return
def query_2_sql(query: Query, table: Table, aggregate=True, split=False, dbms='postgres'):
    preds = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if is_categorical(table.data[col].dtype):
            val = f"\'{val}\'" if not isinstance(val, tuple) else tuple(f"\'{v}\'" for v in val)
        if op == '[]':
            if split:
                preds.append(f"{col} >= {val[0]}")
                preds.append(f"{col} <= {val[1]}")
            else:
                preds.append(f"({col} between {val[0]} and {val[1]})")
        else:
            preds.append(f"{col} {op} {val}")
    return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM \"{table.name}\" WHERE {' AND '.join(preds)}"

def dump_sqls(dataset: str, version: str, workload: str, group: str='test'):
    table = load_table(dataset, version)
    queryset = load_queryset(dataset, workload)
    labels = load_labels(dataset, version, workload)

    with open('test.csv', 'w') as f:
        writer = csv.writer(f)
        for query, label in zip(queryset[group], labels[group]):
            sql = query_2_sql(query, table, aggregate=False, dbms='sqlserver')
            writer.writerow([sql, label.cardinality])