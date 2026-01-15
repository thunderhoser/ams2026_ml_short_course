"""Custom loss functions."""

import keras
import numpy


class CRPS(keras.losses.Loss):
    def __init__(self, function_name, **kwargs):
        """Turns CRPS into loss function.

        :param function_name: Name of function (string).
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

    def call(self, target_tensor, prediction_tensor):
        """Computes CRPS for a single batch.

        B = batch size (number of data samples)
        E = ensemble size (number of members)

        :param prediction_tensor: B-by-E tensor of predicted values.
        :param target_tensor: length-B tensor of actual values.
        :return: loss_value: CRPS (scalar).
        """

        target_tensor = keras.ops.cast(target_tensor, prediction_tensor.dtype)
        target_tensor_2d = keras.ops.expand_dims(target_tensor, axis=-1)

        absolute_error_tensor_2d = keras.ops.abs(
            prediction_tensor - target_tensor_2d
        )
        mae_tensor_1d = keras.ops.mean(absolute_error_tensor_2d, axis=-1)

        abs_pairwise_diff_tensor_3d = keras.ops.abs(
            keras.ops.expand_dims(prediction_tensor, axis=-1) -
            keras.ops.expand_dims(prediction_tensor, axis=-2)
        )
        mapd_tensor_1d = keras.ops.mean(
            abs_pairwise_diff_tensor_3d, axis=(-2, -1)
        )

        crps_tensor_1d = mae_tensor_1d - 0.5 * mapd_tensor_1d
        return keras.ops.mean(crps_tensor_1d)


class MeanSquaredError(keras.losses.Loss):
    def __init__(self, function_name, **kwargs):
        """Turns mean squared error into loss function.

        :param function_name: Name of function (string).
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

    def call(self, target_tensor, prediction_tensor):
        """Computes MSE for a single batch.

        B = batch size (number of data samples)
        E = ensemble size (number of members)

        :param prediction_tensor: B-by-E tensor of predicted values.
        :param target_tensor: length-B tensor of actual values.
        :return: loss_value: MSE (scalar).
        """

        target_tensor = keras.ops.cast(target_tensor, prediction_tensor.dtype)
        prediction_tensor = keras.ops.mean(prediction_tensor, axis=-1)
        squared_error_tensor = (prediction_tensor - target_tensor) ** 2
        return keras.ops.mean(squared_error_tensor)


class CombinedLoss(keras.losses.Loss):
    def __init__(self, function_name, loss_functions, loss_weights, **kwargs):
        """Turns weighted sum of several scores into loss function.

        L = number of individual loss functions

        :param function_name: Name for this combined loss function (string).
        :param loss_functions: length-L list of individual loss functions.
        :param loss_weights: length-L numpy array of weights.
        :param function_name: Name of the combined loss function (string)
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

        assert len(loss_weights.shape) == 1
        assert numpy.all(loss_weights > 0.)
        assert len(loss_functions) == len(loss_weights)

        self.loss_functions = loss_functions
        self.loss_weights = loss_weights

    def call(self, target_tensor, prediction_tensor):
        """Compute combined loss for a single batch.

        :param prediction_tensor: See documentation for `MeanSquaredError`.
        :param target_tensor: Same.
        :return: loss_value: Weighted sum of individual losses (scalar).
        """

        combined_loss = 0.0

        for loss_fn, weight in zip(self.loss_functions, self.loss_weights):
            individual_loss = loss_fn(target_tensor, prediction_tensor)
            combined_loss += weight * individual_loss

        return combined_loss
