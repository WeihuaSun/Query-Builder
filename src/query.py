"""
This is the module for query.
"""
from typing import Dict, NamedTuple, Optional, Tuple, Any
from collections import OrderedDict
from unicodedata import name
from .dataset import Table
from .dtpye import is_categorical
class Query(NamedTuple):
    """predicate of each attritbute are conjunctive"""
    predicates: Dict[str, Optional[Tuple[str, Any]]]
    ncols: int
def new_query(table: Table, ncols) -> Query:
    return Query(predicates=OrderedDict.fromkeys(table.data.columns, None),
                 ncols=ncols)

def query_2_sql(query: Query, table: Table, aggregate=True, split=False):
    """
    Convert queries into SQL query statements.
    
    Args:
        query:A query predicate
        table:The query table
        aggregate:Whether count or not
        split:Decompose the range query
    Return:SQL query
    """
    preds = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if is_categorical(table.data[col].dtype):#categorical£¬place val in single quotes
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