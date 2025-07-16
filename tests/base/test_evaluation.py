import math

import pandas as pd

from sensai import RuleBasedDataFrameTransformer
from sensai.sklearn.sklearn_classification import SkLearnLogisticRegressionVectorClassificationModel
from sensai.sklearn.sklearn_regression import SkLearnLinearRegressionVectorRegressionModel


class DFTDropRows(RuleBasedDataFrameTransformer):
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[5:]


class TestEvaluation:
    def test_evaluation_with_dropped_rows_regression(self, diabetesRegressionTestCase):
        model = SkLearnLinearRegressionVectorRegressionModel()
        model.with_feature_transformers(DFTDropRows())
        diabetesRegressionTestCase.testMinR2(model, -math.inf)

    def test_evaluation_with_dropped_rows_classification(self, irisClassificationTestCase):
        model = SkLearnLogisticRegressionVectorClassificationModel()
        model.with_feature_transformers(DFTDropRows())
        irisClassificationTestCase.testMinAccuracy(model, 0)
