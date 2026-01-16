"""Custom loss functions."""

import keras
import numpy

GREEK_PI = numpy.pi


class CRPS(keras.losses.Loss):
    def __init__(self, function_name, diversity_weight=0., **kwargs):
        """Turns CRPS into loss function.

        :param function_name: Name of function (string).
        :param diversity_weight: Weight for diversity term, which encourages
            each ensemble member to differ among data samples.
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

        assert diversity_weight >= 0.
        self.diversity_weight = diversity_weight

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
        crps_itself = keras.ops.mean(crps_tensor_1d)
        diversity_term = keras.ops.mean(
            keras.ops.std(prediction_tensor, axis=0)
        )

        return crps_itself - self.diversity_weight * diversity_term


class CRPSWeird(keras.losses.Loss):
    def __init__(self, function_name, diversity_weight, **kwargs):
        """Turns CRPS into loss function.

        :param function_name: Name of function (string).
        :param diversity_weight: Weight for diversity term, which encourages
            each ensemble member to differ among data samples.
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

        assert diversity_weight >= 0.
        self.diversity_weight = diversity_weight

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
        crps_itself = keras.ops.mean(crps_tensor_1d)
        diversity_term = keras.ops.mean(
            keras.ops.std(prediction_tensor, axis=0)
        )

        return crps_itself - self.diversity_weight * diversity_term


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


class CRPSGaussian(keras.losses.Loss):
    def __init__(self, function_name='gaussian_crps', **kwargs):
        """Turns parametric CRPS into loss function.

        This parametric CRPS assumes that the target variable follows a Gaussian
        distribution, so the neural net needs to predict only the mean and
        standard deviation of said Gaussian.

        :param function_name: Name of function (string).
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

    def call(self, target_tensor, prediction_tensor):
        """Computes CRPS for a single batch.

        B = batch size (number of data samples)

        :param prediction_tensor: B-by-2 tensor of predicted values, where
            prediction_tensor[:, 0] contains means and prediction_tensor[:, 1]
            contains standard deviations.
        :param target_tensor: length-B tensor of actual values.
        :return: loss_value: CRPS (scalar).
        """

        target_tensor = keras.ops.cast(target_tensor, prediction_tensor.dtype)
        predicted_mean_tensor = prediction_tensor[:, 0]
        predicted_stdev_tensor = keras.ops.maximum(
            prediction_tensor[:, 1], 1e-7
        )

        standardized_error_tensor = (
            (target_tensor - predicted_mean_tensor) / predicted_stdev_tensor
        )
        normal_cdf_tensor = 0.5 * (
            1. +
            keras.ops.erf(standardized_error_tensor / keras.ops.sqrt(2.))
        )
        normal_pdf_tensor = (
            keras.ops.exp(-0.5 * standardized_error_tensor ** 2) /
            keras.ops.sqrt(2. * GREEK_PI)
        )

        first_term_tensor = standardized_error_tensor * (2. * normal_cdf_tensor - 1.)
        second_term_tensor = 2. * normal_pdf_tensor
        third_term_tensor = -1. / keras.ops.sqrt(GREEK_PI)

        crps_tensor_1d = predicted_stdev_tensor * (
            first_term_tensor + second_term_tensor + third_term_tensor
        )
        return keras.ops.mean(crps_tensor_1d)


class MSEGaussian(keras.losses.Loss):
    def __init__(self, function_name, **kwargs):
        """Turns parametric MSE into loss function.

        This parametric MSE assumes that the target variable follows a Gaussian
        distribution, so the neural net needs to predict only the mean and
        standard deviation of said Gaussian.

        :param function_name: Name of function (string).
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

    def call(self, target_tensor, prediction_tensor):
        """Computes MSE for a single batch.

        B = batch size (number of data samples)

        :param prediction_tensor: B-by-2 tensor of predicted values, where
            prediction_tensor[:, 0] contains means and prediction_tensor[:, 1]
            contains standard deviations.
        :param target_tensor: length-B tensor of actual values.
        :return: loss_value: MSE (scalar).
        """

        target_tensor = keras.ops.cast(target_tensor, prediction_tensor.dtype)
        squared_error_tensor = (prediction_tensor[:, 0] - target_tensor) ** 2
        return keras.ops.mean(squared_error_tensor)


class QuantileLoss(keras.losses.Loss):
    def __init__(self, quantile_levels, function_name, **kwargs):
        """Turns quantile loss into loss function.

        :param quantile_levels: 1-D numpy array of quantile levels, all ranging
            from (0, 1).
        :param function_name: Name of function (string).
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)
        
        assert len(quantile_levels.shape) == 1
        assert numpy.all(quantile_levels > 0.)
        assert numpy.all(quantile_levels < 1.)

        self.quantile_levels = quantile_levels

    def call(self, target_tensor, prediction_tensor):
        """Computes quantile loss for a single batch.

        B = batch size (number of data samples)
        Q = number of quantiles

        :param target_tensor: length-B tensor of actual values.
        :param prediction_tensor: B-by-Q tensor of predicted quantiles.
        :return: loss_value: Quantile loss (scalar).
        """

        quantile_level_tensor = keras.ops.convert_to_tensor(
            self.quantile_levels, dtype=prediction_tensor.dtype
        )
        quantile_level_tensor_2d = keras.ops.reshape(
            quantile_level_tensor, (1, self.num_quantiles)
        )

        target_tensor = keras.ops.cast(target_tensor, prediction_tensor.dtype)
        target_tensor_2d = keras.ops.expand_dims(target_tensor, axis=-1)
        target_tensor_2d = keras.ops.repeat(
            target_tensor_2d, self.num_quantiles, axis=-1
        )

        error_tensor_2d = target_tensor_2d - prediction_tensor

        quantile_loss_tensor_2d = keras.ops.maximum(
            quantile_level_tensor_2d * error_tensor_2d,
            (quantile_level_tensor_2d - 1.) * error_tensor_2d
        )

        return keras.ops.mean(quantile_loss_tensor_2d)
