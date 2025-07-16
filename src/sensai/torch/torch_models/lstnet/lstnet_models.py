import collections
import logging
import re
from typing import Optional, Union, Dict, Hashable

import pandas as pd
import torch

from .lstnet_modules import LSTNetwork
from ...torch_base import TorchVectorClassificationModel, VectorTorchModel, ClassificationOutputMode
from ...torch_data import TorchDataSetProviderFromDataUtil, TensorScalerIdentity, TensorScaler, DataUtil
from ...torch_enums import ActivationFunction
from ...torch_opt import NNOptimiserParams

log: logging.Logger = logging.getLogger(__name__)


class LSTNetworkVectorClassificationModel(TorchVectorClassificationModel):
    """
    Classification model for time series data using the LSTNetwork architecture.

    Since the model takes a time series as input, it requires that input data frames to use special naming of columns
    such that the data can be interpreted correctly:
    Each column name must start with an N-digit prefix indicating the time slice the data pertains to (for any fixed N);
    the following suffix shall indicate the name of the actual feature.
    For each N-digit prefix, we must have the same set of suffixes in the list of columns, i.e. we must have the same
    features for each time slice in the input time series.
    """
    def __init__(self, num_input_time_slices, input_dim_per_time_slice, num_classes: Optional[int] = None,
            num_convolutions: int = 100, num_cnn_time_slices: int = 6, hid_rnn: int = 100, skip: int = 0, hid_skip: int = 5,
            hw_window: int = 0, hw_combine: str = "plus", dropout=0.2, output_activation=ActivationFunction.LOG_SOFTMAX, cuda=True,
            nn_optimiser_params: Union[dict, NNOptimiserParams] = None,
            class_weights: Optional[Dict[Hashable, float]] = None) -> None:
        """
        :param num_input_time_slices: the number of input time slices
        :param input_dim_per_time_slice: the dimension of the input data per time slice
        :param num_classes: the number of classes considered by this classification problem; if None, determine from data
        :param num_cnn_time_slices: the number of time slices considered by each convolution (i.e. it is one of the dimensions of the matrix
            used for convolutions, the other dimension being inputDimPerTimeSlice), a.k.a. "Ck"
        :param num_convolutions: the number of separate convolutions to apply, i.e. the number of independent convolution matrices,
            a.k.a "hidC";
            if it is 0, then the entire complex processing path is not applied.
        :param hid_rnn: the number of hidden output dimensions for the RNN stage
        :param skip: the number of time slices to skip for the skip-RNN. If it is 0, then the skip-RNN is not used.
        :param hid_skip: the number of output dimensions of each of the skip parallel RNNs
        :param hw_window: the number of time slices from the end of the input time series to consider as input for the highway component.
            If it is 0, the highway component is not used.
        :param hw_combine: {"plus", "product", "bilinear"} the function with which the highway component's output is combined with the
            complex path's output
        :param dropout: the dropout probability to use during training (dropouts are applied after every major step in the evaluation path)
        :param output_activation: the output activation function
        :param nn_optimiser_params: parameters for NNOptimiser to use for training
        :param class_weights: a mapping from class labels to weights (which will be applied in the default loss evaluator,
            provided that it is not overridden in `nn_optimiser_params`)
        """
        self.num_input_time_slices = num_input_time_slices
        self.input_dim_per_time_slice = input_dim_per_time_slice
        self.num_convolutions = num_convolutions
        self.num_cnn_time_slices = num_cnn_time_slices
        self.hid_rnn = hid_rnn
        self.skip = skip
        self.hid_skip = hid_skip
        self.hw_window = hw_window
        self.hw_combine = hw_combine
        self.dropout = dropout
        self.cuda = cuda
        self.output_activation = output_activation
        self.num_classes = num_classes
        output_mode = ClassificationOutputMode.for_activation_fn(ActivationFunction.torch_function_from_any(output_activation))
        super().__init__(output_mode, self._create_lst_network_model, nn_optimiser_params=nn_optimiser_params, class_weights=class_weights)

    def _create_lst_network_model(self):
        return self._LSTNetworkModel(self)

    class _LSTNetworkModel(VectorTorchModel):
        def __init__(self, parent: "LSTNetworkVectorClassificationModel"):
            super().__init__(parent.cuda)
            self.parent = parent

        def create_torch_module_for_dims(self, input_dim, output_dim):
            p = self.parent
            expected_input_dim = p.num_input_time_slices * p.input_dim_per_time_slice
            if expected_input_dim != input_dim:
                raise ValueError(f"Unexpected input size {input_dim}, expected {self.inputDim}")
            if p.num_classes is None:
                output_dim_per_time_slice = output_dim
            else:
                output_dim_per_time_slice = p.num_classes
                if p.num_classes != output_dim:
                    raise ValueError(f"Unexpected output dim {output_dim}, expected {p.num_classes}")
            return LSTNetwork(num_input_time_slices=p.num_input_time_slices,
                input_dim_per_time_slice=p.input_dim_per_time_slice,
                num_output_time_slices=1,
                output_dim_per_time_slice=output_dim_per_time_slice,
                num_convolutions=p.num_convolutions,
                num_cnn_time_slices=p.num_cnn_time_slices,
                hid_rnn=p.hid_rnn,
                hw_window=p.hw_window,
                hw_combine=p.hw_combine,
                dropout=p.dropout,
                output_activation=p.output_activation,
                skip=p.skip,
                hid_skip=p.hid_skip,
                mode=LSTNetwork.Mode.CLASSIFICATION)

    def _create_data_set_provider(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> TorchDataSetProviderFromDataUtil:
        if self.num_classes is None:
            self.num_classes = len(self._labels)
        elif self.num_classes != len(self._labels):
            raise ValueError(f"Output dimension {self.num_classes} per time time slice was specified, while the training data contains "
                             f"{len(self._labels)} classes")
        return TorchDataSetProviderFromDataUtil(self.DataUtil(inputs, outputs, self.num_classes), self.cuda)

    def _predict_outputs_for_input_data_frame(self, inputs: pd.DataFrame) -> torch.Tensor:
        log.info(f"Predicting outputs for {len(inputs)} inputs")
        result = super()._predict_outputs_for_input_data_frame(inputs)
        return result.squeeze(2)

    def _compute_model_inputs(self, x: pd.DataFrame, y: pd.DataFrame = None, fit=False) -> pd.DataFrame:
        x = super()._compute_model_inputs(x, y=y, fit=fit)

        # sort input data frame columns by name
        x = x[sorted(x.columns)]

        # check input column name format and consistency
        col_name_regex = re.compile(r"(\d+).+")
        cols_by_time_slice = collections.defaultdict(list)
        num_digits = None
        for colName in x.columns:
            match = col_name_regex.fullmatch(colName)
            if not match:
                raise ValueError(f"Column name '{colName}' does not match the required format (N-digit prefix indicating the time slice "
                                 f"order followed by feature name; for any fixed N); columns={list(x.columns)}")
            time_slice = match.group(1)
            if num_digits is None:
                num_digits = len(time_slice)
            elif num_digits != len(time_slice):
                raise ValueError(f"Inconsistent number of digits in column names: Got {num_digits} leading digits for one feature and "
                                 f"{len(time_slice)} for another; columns={list(x.columns)}")
            cols_by_time_slice[time_slice].append(colName[num_digits:])
        reference_cols = None
        for time_slice, cols in cols_by_time_slice.items():
            if reference_cols is None:
                reference_cols = cols
            elif reference_cols != cols:
                raise ValueError(f"Inconsistent features across time slices: Got suffixes {cols} for one time slice and {reference_cols} "
                                 f"for another; columns={list(x.columns)}")

        return x

    class DataUtil(DataUtil):
        def __init__(self, x_data: pd.DataFrame, y_data: pd.DataFrame, num_classes):
            self.y_data = y_data
            self.x_data = x_data
            self.numClasses = num_classes
            self.scaler = TensorScalerIdentity()

        def input_dim(self):
            return len(self.x_data.columns)

        def model_output_dim(self) -> int:
            return self.numClasses

        def split_into_tensors(self, fractional_size_of_first_set):
            split_index = round(fractional_size_of_first_set * len(self.y_data))
            y1, x1 = self.get_input_output_pair(self.y_data[:split_index], self.x_data[:split_index])
            y2, x2 = self.get_input_output_pair(self.y_data[split_index:], self.x_data[split_index:])
            return (x1, y1), (x2, y2)

        def get_input_output_pair(self, output, input):
            y = torch.tensor(output.values).long()
            x = torch.tensor(input.values).float()
            return y, x

        def get_output_tensor_scaler(self) -> TensorScaler:
            return self.scaler

        def get_input_tensor_scaler(self) -> TensorScaler:
            return self.scaler
