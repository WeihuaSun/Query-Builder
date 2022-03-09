from sqlalchemy import false
from tomlkit import string
import generator
import random
import csv
import numpy as np
import argparse
from time import time
from dataset import load_table,csv_to_pkl
from generator import QueryGenerator
from query import Query
from dataset import Table
from typing import Dict, Any
from ast import literal_eval
from dtpye import is_categorical
def generate_queries( seed: int, dataset: str, name: str, number:int ,params: Dict[str, Dict[str, Any]]
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    attr_funcs = {getattr(generator, f"asf_{a}"): v for a, v in params['attr'].items()}
    center_funcs = {getattr(generator, f"csf_{c}"): v for c, v in params['center'].items()}
    width_funcs = {getattr(generator, f"wsf_{w}"): v for w, v in params['width'].items()}
    csv_to_pkl(dataset)
    table = load_table(dataset)
    qgen = QueryGenerator(
            table=table,
            attr=attr_funcs,
            center=center_funcs,
            width=width_funcs,
            attr_params=params.get('attr_params') or {},
            center_params=params.get('center_params') or {},
            width_params=params.get('width_params') or {})

    num =number
    queries = []
    for i in range(num):
        queries.append(qgen.generate())
    queryset = queries
    dump_sqls(queryset,table)

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
    return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM {table.name} WHERE {' AND '.join(preds)};"
     


def dump_sqls(queryset,table):
    with open('test.sql', 'w') as f:
        writer = csv.writer(f)
        for query in queryset:
            sql = query_2_sql(query, table, aggregate=False)
            writer.writerow([sql])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                    default=None,
                    type = int,
                    required = False,
                    help="Random seed"          
    )
    parser.add_argument('--datasets',
                    default="census",
                    type = str,
                    required = False,
                    help="datasets name"          
    )
    parser.add_argument('--name',
                    default="newquery",
                    type = str,
                    required = False,
                    help="name"          
    )
    parser.add_argument('--number',
                    default=10000,
                    type = int,
                    required = False,
                    help="number of queries"          
    )
    parser.add_argument('--params',
                    default="{'attr': {'pred_number': 1.0}, \
                            'center': {'distribution': 0.9, 'vocab_ood': 0.1}, \
                            'width': {'uniform': 0.5, 'exponential': 0.5}}",
                    required=False,
                    )
    args = parser.parse_args()
    if args.seed is None:
        seed = int(time())
    else:
        seed = int(args.seed)
    generate_queries(
        seed = seed,
        dataset=args.datasets,
        name=args.name,
        number=args.number,
        params = literal_eval(args.params)
    )
