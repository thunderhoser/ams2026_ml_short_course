"""Custom loss functions."""

import keras


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
