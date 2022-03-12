"""
Usage:
  python -m main  [--s <seed>] [--d <dataset>]  [-q <query>] [--params <params>] 
  params:"  {'attr': {'pred_number': p1[, 'fun2': p2]}, \
            'center': {'distribution': p1[, 'fun2': p2]}, \
            'width': {'uniform': p1[, 'fun2': p2]},\
            'attr_params':{'whitelist':['age','workclass']},\
            'center_params':{},
            'width_params':{},
            }  "
 
Options:
  --s <seed>                Random seed.
  --d <dataset>             The input dataset [default: census].
  --q <query>               Name of the sqls [default: newquery].
  --n <number>              The number of queries to be generated[default: 10000].
  --params <params>         Parameters that are needed.
"""

from src import generator
import random
import csv
import numpy as np
import argparse
from time import time
from src.dataset import load_table,csv_to_pkl
from src.generator import QueryGenerator
from src.query import query_2_sql
from typing import Dict, Any
from ast import literal_eval
from src.constants import OUTPUT_ROOT

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

    num = number
    queries = []
    for i in range(num):
        queries.append(qgen.generate())
    dump_sqls(queries,table,name)


def dump_sqls(queryset,table,name):
    """
    Dump SQLs to disk
    """
    with open(OUTPUT_ROOT/f"{name}.sql", 'w',newline="") as f:
        writer = csv.writer(f)
        for query in queryset:
            sql = query_2_sql(query, table, aggregate=False)
            writer.writerow([sql])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--s',
                    default=int(time()),
                    type = int,
                    required = False,
                    help="Random seed"          
    )
    parser.add_argument('--d',
                    default="census",
                    type = str,
                    required = False,
                    help="Datasets name"          
    )
    parser.add_argument('--q',
                    default="newquery",
                    type = str,
                    required = False,
                    help="The name of the generated query file"          
    )
    parser.add_argument('--n',
                    default=10000,
                    type = int,
                    required = False,
                    help="The number of queries to be generated"          
    )
    parser.add_argument('--params',
                    default="{'attr': {'pred_number': 1.0}, \
                            'center': {'distribution': 0.9, 'vocab_ood': 0.1}, \
                            'width': {'uniform': 0.5, 'exponential': 0.5}\
                            }",
                            
                    required=False,
                    )
    args = parser.parse_args()

    generate_queries(
        seed = args.s,
        dataset=args.d,
        name=args.q,
        number=args.n,
        params = literal_eval(args.params)
    )
