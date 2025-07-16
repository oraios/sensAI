from typing import Optional, List, Self, TYPE_CHECKING, Callable

import pandas as pd

from sensai.evaluation.eval_stats import RegressionEvalStats
from sensai.util.pandas import query_data_frame
from sensai.vector_model import get_predicted_var_name

if TYPE_CHECKING:
    from sensai.evaluation import VectorRegressionModelEvaluationData


class ResultSet:
    """
    A result set which is designed for interactive result inspection (e.g. in an iPython notebook).
    An instance can, for example, be created with a data frame as returned by VectorRegressionModelEvaluationData.to_data_frame
    and subsequently be applied to interactively analyse the results.

    The class is designed to be subclassed, such that, in particular, method `_show_df` can be
    overridden to display meaningful information (use case-specific) in the notebook environment.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _create_result_set(self, df: pd.DataFrame) -> Self:
        """
        Creates a new result set for the given data frame

        :param df: the data frame
        :return: the result set
        """
        return self.__class__(df)

    def query(self, sql: str) -> Self:
        """
        Queries the result set with the given condition specified in SQL syntax.

        NOTE: Requires duckdb to be installed.

        :param sql: an SQL query starting with the WHERE clause (excluding the 'where' keyword itself)
        :return: the result set corresponding to the query
        """
        result_df = query_data_frame(self.df, sql)
        return self._create_result_set(result_df)

    def show(self, first: Optional[int] = None, sample: Optional[int] = None) -> None:
        """
        Shows all or some of the result set's contents.

        :param first: if not None, show this many rows from the start of the result set
        :param sample: if not None, sample this many rows from the result set to be shown
        """
        df = self.df
        if first is not None:
            df = df.iloc[:first]
        if sample is not None:
            df = df.sample(sample)
        self._show_df(df)

    def _show_df(self, df: pd.DataFrame):
        print(df.to_string())

    @staticmethod
    def col_name_predicted(predicted_var_name: str):
        return f"{predicted_var_name}_predicted"

    @staticmethod
    def col_name_ground_truth(predicted_var_name: str):
        return f"{predicted_var_name}_true"


class RegressionResultSet(ResultSet):
    def __init__(self, df: pd.DataFrame, predicted_var_names: List[str]):
        super().__init__(df)
        self.predicted_var_names = predicted_var_names

    @classmethod
    def from_regression_eval_data(cls, eval_data: "VectorRegressionModelEvaluationData", modify_input_df: bool = False,
            output_col_name_override: Optional[str] = None,
            regression_result_set_factory: Callable[[pd.DataFrame, List[str]], "RegressionResultSet"] = None) \
            -> "RegressionResultSet":
        df = eval_data.to_data_frame(modify_input_df=modify_input_df, output_col_name_override=output_col_name_override)
        if output_col_name_override:
            predicted_var_names = [output_col_name_override]
        else:
            predicted_var_names = eval_data.predicted_var_names

        def default_factory(data_frame: pd.DataFrame, var_names: List[str]):
            return cls(data_frame, var_names)

        if regression_result_set_factory is None:
            regression_result_set_factory = default_factory

        return regression_result_set_factory(df, predicted_var_names)

    def _create_result_set(self, df: pd.DataFrame):
        return self.__class__(df, self.predicted_var_names)

    @staticmethod
    def col_name_error(predicted_var_name: str):
        return f"{predicted_var_name}_error"

    @staticmethod
    def col_name_abs_error(predicted_var_name: str):
        return f"{predicted_var_name}_abs_error"

    def eval_stats(self, predicted_var_name: Optional[str] = None):
        """
        Creates the evaluation stats object for this result object, which can be used to compute metrics
        or to create plots.

        :param predicted_var_name: the name of the predicted variable for which to create the object;
            can be None if there is but a single variable
        :return: the evaluation stats object
        """
        predicted_var_name = get_predicted_var_name(predicted_var_name, self.predicted_var_names)
        return RegressionEvalStats(y_predicted=self.df[self.col_name_predicted(predicted_var_name)],
            y_true=self.df[self.col_name_ground_truth(predicted_var_name)])
