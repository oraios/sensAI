import json
from numbers import Integral, Real
from typing import Optional

import numpy as np
import pandas as pd

SNAPSHOT_FLOAT_DECIMALS_DEFAULT = 6
"""The default number of float decimal places to consider when converting to a snapshot. 
Overwrite this constant in user code if you want to change the default for all calls to `snapshot_compatible`"""
SNAPSHOT_SIGNIFICANT_DIGITS_DEFAULT = 12
"""The (maximum) number of significant digits to consider when converting to a snapshot. 
Overwrite this constant in user code if you want to change the default for all calls to `snapshot_compatible`"""


def snapshot_compatible(obj, float_decimals: Optional[int] = None, significant_digits: Optional[int] = None):
    """
    Renders an object snapshot-compatible by appropriately converting nested types and reducing float precision to a level
    that is likely to not cause problems when testing snapshots for equivalence on different platforms.
    Works with many standard python, numpy and pandas objects, including arrays and data frames.

    :param obj: the object to convert
    :param float_decimals: the number of float decimal places to consider, by default it is 6 (see `SNAPSHOT_FLOAT_DECIMALS_DEFAULT`)
    :param significant_digits: the (maximum) number of significant digits to consider, by default it is 12 (see `SNAPSHOT_SIGNIFICANT_DIGITS_DEFAULT`)
    :return: the converted object
    """
    # we have to do it with None if we want to let users overwrite this,
    # since python binds values to kwargs immediately
    if float_decimals is None:
        float_decimals = SNAPSHOT_FLOAT_DECIMALS_DEFAULT
    if significant_digits is None:
        significant_digits = SNAPSHOT_SIGNIFICANT_DIGITS_DEFAULT

    result = json.loads(json.dumps(obj, default=json_mapper))
    return convert_floats(result, float_decimals, significant_digits)


def reduce_float_precision(f, decimals, significant_digits):
    return float(format(float(format(f, '.%df' % decimals)), ".%dg" % significant_digits))


def convert_floats(o, float_decimals, significant_digits):
    if type(o) == list:
        return [convert_floats(x, float_decimals, significant_digits) for x in o]
    elif type(o) == dict:
        return {key: convert_floats(value, float_decimals, significant_digits) for (key, value) in o.items()}
    elif type(o) == float:
        return reduce_float_precision(o, float_decimals, significant_digits)
    else:
        return o


def json_mapper(o):
    """
    Maps the given data object to a representation that is JSON-compatible.
    Currently, the supported object types include, in particular, numpy arrays as well as pandas Series and DataFrames.

    :param o: the object to convert
    :return: the converted object
    """
    if isinstance(o, pd.DataFrame):
        if isinstance(o.index, pd.DatetimeIndex):
            o.index = o.index.astype('int64').tolist()
        return o.to_dict()
    if isinstance(o, pd.Series):
        return o.values.tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, list):
        return o
    if isinstance(o, tuple):
        return list(o)
    # without casting to int or float we get weird recursion related errors
    if isinstance(o, Integral):
        return int(o)
    if isinstance(o, Real):
        return float(o)
    else:
        return o.__dict__
