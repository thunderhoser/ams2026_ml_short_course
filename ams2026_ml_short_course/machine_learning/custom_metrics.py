"""Custom metrics."""

import keras


class MeanSquaredError(keras.metrics.Metric):
    def __init__(self, function_name, **kwargs):
        """Turns mean squared error into metric.

        :param function_name: Name of function (string).
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

        self.total_weighted_mse = self.add_weight(
            name='total_weighted_mse', initializer='zeros'
        )
        self.total_weight = self.add_weight(
            name='total_weight', initializer='zeros'
        )

    def update_state(self, target_tensor, prediction_tensor,
                     sample_weight=None):
        """Updates MSE.

        B = batch size (number of data samples)
        E = ensemble size (number of members)

        :param prediction_tensor: B-by-E tensor of predicted values.
        :param target_tensor: length-B tensor of actual values.
        :param sample_weight: Leave this alone.
        """

        target_tensor = keras.ops.cast(target_tensor, prediction_tensor.dtype)
        target_tensor_2d = keras.ops.expand_dims(target_tensor, axis=-1)
        prediction_tensor_2d = target_tensor_2d

        squared_error_tensor_2d = (prediction_tensor_2d - target_tensor_2d) ** 2
        mse_tensor_1d = keras.ops.mean(squared_error_tensor_2d, axis=-1)

        self.total_weighted_mse.assign_add(keras.ops.mean(mse_tensor_1d))
        self.total_weight.assign_add(1.)

    def result(self):
        """Computes final MSE.

        :return: mean_squared_error: MSE (a scalar value).
        """

        return (
            self.total_weighted_mse /
            keras.ops.maximum(self.total_weight, 1e-7)
        )

    def reset_state(self):
        """Resets values between epochs."""

        self.total_weighted_mse.assign(0.)
        self.total_weight.assign(0.)
