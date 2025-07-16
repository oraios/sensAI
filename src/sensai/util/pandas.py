import logging
import re
from abc import ABC, abstractmethod
from copy import copy
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sensai.util import mark_used
from sensai.util.pickle import getstate
from sensai.util.string import ToStringMixin

log = logging.getLogger(__name__)


class DataFrameColumnChangeTracker:
    """
    A simple class for keeping track of changes in columns between an initial data frame and some other data frame
    (usually the result of some transformations performed on the initial one).

    Example:

    >>> from sensai.util.pandas import DataFrameColumnChangeTracker
    >>> import pandas as pd

    >>> df = pd.DataFrame({"bar": [1, 2]})
    >>> columnChangeTracker = DataFrameColumnChangeTracker(df)
    >>> df["foo"] = [4, 5]
    >>> columnChangeTracker.track_change(df)
    >>> columnChangeTracker.get_removed_columns()
    set()
    >>> columnChangeTracker.get_added_columns()
    {'foo'}
    """
    def __init__(self, initial_df: pd.DataFrame):
        self.initialColumns = copy(initial_df.columns)
        self.final_columns = None

    def track_change(self, changed_df: pd.DataFrame):
        self.final_columns = copy(changed_df.columns)

    def get_removed_columns(self):
        self.assert_change_was_tracked()
        return set(self.initialColumns).difference(self.final_columns)

    def get_added_columns(self):
        """
        Returns the columns in the last entry of the history that were not present the first one
        """
        self.assert_change_was_tracked()
        return set(self.final_columns).difference(self.initialColumns)

    def column_change_string(self):
        """
        Returns a string representation of the change
        """
        self.assert_change_was_tracked()
        if list(self.initialColumns) == list(self.final_columns):
            return "none"
        removed_cols, added_cols = self.get_removed_columns(), self.get_added_columns()
        if removed_cols == added_cols == set():
            return f"reordered {list(self.final_columns)}"

        return f"added={list(added_cols)}, removed={list(removed_cols)}"

    def assert_change_was_tracked(self):
        if self.final_columns is None:
            raise Exception(f"No change was tracked yet. "
                            f"Did you forget to call trackChange on the resulting data frame?")


class ColumnMatcher(ToStringMixin, ABC):
    @abstractmethod
    def matches(self, name: str) -> bool:
        pass


class ColumnMatcherCollection:
    def __init__(self, matchers: Sequence[Union[str, ColumnMatcher]]):
        self.matchers = []
        for m in matchers:
            if isinstance(m, str):
                self.matchers.append(ColumnName(m))
            elif isinstance(m, ColumnMatcher):
                self.matchers.append(m)
            else:
                raise ValueError(f"{m} is not a string or ColumnMatcher")

    def matching_columns(self, columns: Sequence[str], require_all_matchers_applied: bool = True) -> List[str]:
        """
        :param columns: the columns to check
        :param require_all_matchers_applied: whether to require all matchers to match at least one column and raise
            an exception otherwise
        :return: the subset of the given columns that match at least one of this collection's matchers
        """
        result = []
        for m in self.matchers:
            found_match = False
            for c in columns:
                if m.matches(c):
                    result.append(c)
                    found_match = True
                    break
            if not found_match and require_all_matchers_applied:
                raise ValueError(f"{m} did not match any columns in {columns}")
        return result

    def not_matching_columns(self, columns: Sequence[str]) -> List[str]:
        """
        :param columns: the columns to check
        :return: the subset of the given columns that do not match any of this collection's matchers
        """
        matching_columns = set(self.matching_columns(columns, require_all_matchers_applied=False))
        return [c for c in columns if c not in matching_columns]


class ColumnName(ColumnMatcher):
    def __init__(self, name: str):
        self.name = name

    def matches(self, name: str) -> bool:
        return self.name == name


class ColumnRegex(ColumnMatcher):
    def __init__(self, regex: str, flags: int = 0):
        self.regex = regex
        self.flags = flags
        self._pattern = None

    def __getstate__(self):
        return getstate(ColumnRegex, self, transient_properties=["_pattern"])

    def _tostring_exclude_private(self) -> bool:
        return True

    def _get_pattern(self) -> re.Pattern:
        if self._pattern is None:
            self._pattern = re.compile(self.regex, flags=self.flags)
        return self._pattern

    def matches(self, name: str) -> bool:
        return self._get_pattern().fullmatch(name) is not None


def extract_array(df: pd.DataFrame, dtype=None):
    """
    Extracts array from data frame. It is expected that each row corresponds to a data point and
    each column corresponds to a "channel". Moreover, all entries are expected to be arrays of the same shape
    (or scalars or sequences of the same length). We will refer to that shape as tensorShape.

    The output will be of shape `(N_rows, N_columns, *tensorShape)`. Thus, `N_rows` can be interpreted as dataset length
    (or batch size, if a single batch is passed) and N_columns can be interpreted as number of channels.
    Empty dimensions will be stripped, thus if the data frame has only one column, the array will have shape
    `(N_rows, *tensorShape)`.
    E.g. an image with three channels could equally be passed as data frame of the type


    +------------------+------------------+------------------+
    | R                | G                | B                |
    +==================+==================+==================+
    | channel          | channel          | channel          |
    +------------------+------------------+------------------+
    | channel          | channel          | channel          |
    +------------------+------------------+------------------+
    | ...              | ...              | ...              |
    +------------------+------------------+------------------+

    or as data frame of type

    +------------------+
    | image            |
    +==================+
    | RGB-array        |
    +------------------+
    | RGB-array        |
    +------------------+
    | ...              |
    +------------------+

    In both cases the returned array will have shape `(N_images, 3, width, height)`

    :param df: data frame where each entry is an array of shape tensorShape
    :param dtype: if not None, convert the array's data type to this type (string or numpy dtype)
    :return: array of shape `(N_rows, N_columns, *tensorShape)` with stripped empty dimensions
    """
    log.debug(f"Stacking tensors of shape {np.array(df.iloc[0, 0]).shape}")
    try:
        # This compact way of extracting the array causes dtypes to be modified,
        #    arr = np.stack(df.apply(np.stack, axis=1)).squeeze()
        # so we use this numpy-only alternative:
        arr = df.values
        if arr.shape[1] > 1:
            arr = np.stack([np.stack(arr[i]) for i in range(arr.shape[0])])
        else:
            arr = np.stack(arr[:, 0])
        # For the case where there is only one row, the old implementation above removed the first dimension,
        # so we do the same, even though it seems odd to do so (potential problem for batch size 1)
        # TODO: remove this behavior
        if arr.shape[0] == 1:
            arr = arr[0]
    except ValueError:
        raise ValueError(f"No array can be extracted from frame of length {len(df)} with columns {list(df.columns)}. "
                         f"Make sure that all entries have the same shape")
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def remove_duplicate_index_entries(df: pd.DataFrame):
    """
    Removes successive duplicate index entries by keeping only the first occurrence for every duplicate index element.

    :param df: the data frame, which is assumed to have a sorted index
    :return: the (modified) data frame with duplicate index entries removed
    """
    keep = [True]
    prev_item = df.index[0]
    for item in df.index[1:]:
        keep.append(item != prev_item)
        prev_item = item
    return df[keep]


def query_data_frame(df: pd.DataFrame, sql: str):
    """
    Queries the given data frame with the given condition specified in SQL syntax.

    NOTE: Requires duckdb to be installed.

    :param df: the data frame to query
    :param sql: an SQL query starting with the WHERE clause (excluding the 'where' keyword itself)
    :return: the filtered/transformed data frame
    """
    import duckdb

    NUM_TYPE_INFERENCE_ROWS = 100

    def is_supported_object_col(col_name: str):
        supported_type_set = set()
        contains_unsupported_types = False
        # check the first N values
        for value in df[col_name].iloc[:NUM_TYPE_INFERENCE_ROWS]:
            if isinstance(value, str):
                supported_type_set.add(str)
            elif value is None:
                pass
            else:
                contains_unsupported_types = True
        return not contains_unsupported_types and len(supported_type_set) == 1

    # determine which columns are object columns that are unsupported by duckdb and would raise errors
    # if they remained in the data frame that is queried
    added_index_col = "__sensai_resultset_index__"
    original_columns = df.columns
    object_columns = list(df.dtypes[df.dtypes == object].index)
    object_columns = [c for c in object_columns if not is_supported_object_col(c)]

    # add an artificial index which we will use to identify the rows for object column reconstruction
    df[added_index_col] = np.arange(len(df))

    try:
        # remove the object columns from the data frame but save them for subsequent reconstruction
        objects_df = df[object_columns + [added_index_col]]
        query_df = df.drop(columns=object_columns)
        mark_used(query_df)

        # apply query with reduced df
        result_df = duckdb.query(f"select * from query_df where {sql}").to_df()

        # restore object columns in result
        objects_df.set_index(added_index_col, drop=True, inplace=True)
        result_df.set_index(added_index_col, drop=True, inplace=True)
        result_objects_df = objects_df.loc[result_df.index]
        assert len(result_df) == len(result_objects_df)
        full_result_df = pd.concat([result_df, result_objects_df], axis=1)
        full_result_df = full_result_df[original_columns]

    finally:
        # clean up
        df.drop(columns=added_index_col, inplace=True)

    return full_result_df


class SeriesInterpolation(ABC):
    def interpolate(self, series: pd.Series, inplace: bool = False) -> Optional[pd.Series]:
        if not inplace:
            series = series.copy()
        self._interpolate_in_place(series)
        return series if not inplace else None

    @abstractmethod
    def _interpolate_in_place(self, series: pd.Series) -> None:
        pass

    def interpolate_all_with_combined_index(self, series_list: List[pd.Series]) -> List[pd.Series]:
        """
        Interpolates the given series using the combined index of all series.

        :param series_list: the list of series to interpolate
        :return: a list of corresponding interpolated series, each having the same index
        """
        # determine common index and
        index_elements = set()
        for series in series_list:
            index_elements.update(series.index)
        common_index = sorted(index_elements)

        # reindex, filling the gaps via interpolation
        interpolated_series_list = []
        for series in series_list:
            series = series.copy()
            series = series.reindex(common_index, method=None)
            self.interpolate(series, inplace=True)
            interpolated_series_list.append(series)

        return interpolated_series_list


class SeriesInterpolationLinearIndex(SeriesInterpolation):
    def __init__(self, ffill: bool = False, bfill: bool = False):
        """
        :param ffill: whether to fill any N/A values at the end of the series with the last valid observation
        :param bfill: whether to fill any N/A values at the start of the series with the first valid observation
        """
        self.ffill = ffill
        self.bfill = bfill

    def _interpolate_in_place(self, series: pd.Series) -> None:
        series.interpolate(method="index", inplace=True)
        if self.ffill:
            series.interpolate(method="ffill", limit_direction="forward")
        if self.bfill:
            series.interpolate(method="bfill", limit_direction="backward")


class SeriesInterpolationRepeatPreceding(SeriesInterpolation):
    def __init__(self, bfill: bool = False):
        """
        :param bfill: whether to fill any N/A values at the start of the series with the first valid observation
        """
        self.bfill = bfill

    def _interpolate_in_place(self, series: pd.Series) -> None:
        series.interpolate(method="pad", limit_direction="forward", inplace=True)
        if self.bfill:
            series.interpolate(method="bfill", limit_direction="backward")


def average_series(series_list: List[pd.Series], interpolation: SeriesInterpolation) -> pd.Series:
    interpolated_series_list = interpolation.interpolate_all_with_combined_index(series_list)
    return sum(interpolated_series_list) / len(interpolated_series_list)  # type: ignore
