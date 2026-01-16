"""Custom metrics."""

import numpy
import keras

GREEK_PI = numpy.pi


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
        prediction_tensor_2d = prediction_tensor

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


class MSEGaussian(keras.metrics.Metric):
    def __init__(self, function_name, **kwargs):
        """Turns parametric MSE into metric.

        This parametric MSE assumes that the target variable follows a Gaussian
        distribution, so the neural net needs to predict only the mean and
        standard deviation of said Gaussian.

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

        :param prediction_tensor: B-by-2 tensor of predicted values, where
            prediction_tensor[:, 0] contains means and prediction_tensor[:, 1]
            contains standard deviations.
        :param target_tensor: length-B tensor of actual values.
        :param sample_weight: Leave this alone.
        """

        target_tensor = keras.ops.cast(target_tensor, prediction_tensor.dtype)
        squared_error_tensor = (prediction_tensor[:, 0] - target_tensor) ** 2

        self.total_weighted_mse.assign_add(keras.ops.mean(squared_error_tensor))
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


class MeanAbsoluteError(keras.metrics.Metric):
    def __init__(self, function_name, **kwargs):
        """Turns mean absolute error into metric.

        :param function_name: Name of function (string).
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

        self.total_weighted_mae = self.add_weight(
            name='total_weighted_mae', initializer='zeros'
        )
        self.total_weight = self.add_weight(
            name='total_weight', initializer='zeros'
        )

    def update_state(self, target_tensor, prediction_tensor,
                     sample_weight=None):
        """Updates MAE.

        B = batch size (number of data samples)
        E = ensemble size (number of members)

        :param prediction_tensor: B-by-E tensor of predicted values.
        :param target_tensor: length-B tensor of actual values.
        :param sample_weight: Leave this alone.
        """

        target_tensor = keras.ops.cast(target_tensor, prediction_tensor.dtype)
        target_tensor_2d = keras.ops.expand_dims(target_tensor, axis=-1)
        prediction_tensor_2d = prediction_tensor

        absolute_error_tensor_2d = keras.ops.abs(
            prediction_tensor_2d - target_tensor_2d
        )
        mae_tensor_1d = keras.ops.mean(absolute_error_tensor_2d, axis=-1)

        self.total_weighted_mae.assign_add(keras.ops.mean(mae_tensor_1d))
        self.total_weight.assign_add(1.)

    def result(self):
        """Computes final MAE.

        :return: mean_absolute_error: MAE (a scalar value).
        """

        return (
            self.total_weighted_mae /
            keras.ops.maximum(self.total_weight, 1e-7)
        )

    def reset_state(self):
        """Resets values between epochs."""

        self.total_weighted_mae.assign(0.)
        self.total_weight.assign(0.)


class MAEGaussian(keras.metrics.Metric):
    def __init__(self, function_name, **kwargs):
        """Turns parametric MAE into metric.

        This parametric MAE assumes that the target variable follows a Gaussian
        distribution, so the neural net needs to predict only the mean and
        standard deviation of said Gaussian.

        :param function_name: Name of function (string).
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

        self.total_weighted_mae = self.add_weight(
            name='total_weighted_mae', initializer='zeros'
        )
        self.total_weight = self.add_weight(
            name='total_weight', initializer='zeros'
        )

    def update_state(self, target_tensor, prediction_tensor,
                     sample_weight=None):
        """Updates MAE.

        B = batch size (number of data samples)

        :param prediction_tensor: B-by-2 tensor of predicted values, where
            prediction_tensor[:, 0] contains means and prediction_tensor[:, 1]
            contains standard deviations.
        :param target_tensor: length-B tensor of actual values.
        :param sample_weight: Leave this alone.
        """

        target_tensor = keras.ops.cast(target_tensor, prediction_tensor.dtype)
        absolute_error_tensor = keras.ops.abs(
            prediction_tensor[:, 0] - target_tensor
        )

        self.total_weighted_mae.assign_add(
            keras.ops.mean(absolute_error_tensor)
        )
        self.total_weight.assign_add(1.)

    def result(self):
        """Computes final MAE.

        :return: mean_absolute_error: MAE (a scalar value).
        """

        return (
            self.total_weighted_mae /
            keras.ops.maximum(self.total_weight, 1e-7)
        )

    def reset_state(self):
        """Resets values between epochs."""

        self.total_weighted_mae.assign(0.)
        self.total_weight.assign(0.)


class MeanAbsPairwiseDiff(keras.metrics.Metric):
    def __init__(self, function_name, **kwargs):
        """Turns mean absolute pairwise difference into metric.

        :param function_name: Name of function (string).
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

        self.total_weighted_mapd = self.add_weight(
            name='total_weighted_mapd', initializer='zeros'
        )
        self.total_weight = self.add_weight(
            name='total_weight', initializer='zeros'
        )

    def update_state(self, target_tensor, prediction_tensor,
                     sample_weight=None):
        """Updates MAPD.

        B = batch size (number of data samples)
        E = ensemble size (number of members)

        :param prediction_tensor: B-by-E tensor of predicted values.
        :param target_tensor: length-B tensor of actual values.
        :param sample_weight: Leave this alone.
        """

        abs_pairwise_diff_tensor_3d = keras.ops.abs(
            keras.ops.expand_dims(prediction_tensor, axis=-1) -
            keras.ops.expand_dims(prediction_tensor, axis=-2)
        )
        mapd_tensor_1d = keras.ops.mean(
            abs_pairwise_diff_tensor_3d, axis=(-2, -1)
        )

        self.total_weighted_mapd.assign_add(keras.ops.mean(mapd_tensor_1d))
        self.total_weight.assign_add(1.)

    def result(self):
        """Computes final MAPD.

        :return: mapd_value: MAPD (a scalar value).
        """

        return (
            self.total_weighted_mapd /
            keras.ops.maximum(self.total_weight, 1e-7)
        )

    def reset_state(self):
        """Resets values between epochs."""

        self.total_weighted_mapd.assign(0.)
        self.total_weight.assign(0.)


class CRPS(keras.metrics.Metric):
    def __init__(self, function_name, diversity_weight=0., **kwargs):
        """Turns CRPS into metric.

        :param function_name: Name of function (string).
        :param diversity_weight: Weight for diversity term, which encourages
            each ensemble member to differ among data samples.
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

        assert diversity_weight >= 0.
        self.diversity_weight = diversity_weight

        self.total_weighted_crps = self.add_weight(
            name='total_weighted_crps', initializer='zeros'
        )
        self.total_weight = self.add_weight(
            name='total_weight', initializer='zeros'
        )

    def update_state(self, target_tensor, prediction_tensor,
                     sample_weight=None):
        """Updates CRPS.

        B = batch size (number of data samples)
        E = ensemble size (number of members)

        :param prediction_tensor: B-by-E tensor of predicted values.
        :param target_tensor: length-B tensor of actual values.
        :param sample_weight: Leave this alone.
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

        self.total_weighted_crps.assign_add(
            crps_itself - self.diversity_weight * diversity_term
        )
        self.total_weight.assign_add(1.)

    def result(self):
        """Computes final CRPS.

        :return: crps_value: CRPS (a scalar value).
        """

        return (
            self.total_weighted_crps /
            keras.ops.maximum(self.total_weight, 1e-7)
        )

    def reset_state(self):
        """Resets values between epochs."""

        self.total_weighted_crps.assign(0.)
        self.total_weight.assign(0.)


class CRPSGaussian(keras.metrics.Metric):
    def __init__(self, function_name, **kwargs):
        """Turns parametric CRPS into metric.

        This parametric CRPS assumes that the target variable follows a Gaussian
        distribution, so the neural net needs to predict only the mean and
        standard deviation of said Gaussian.

        :param function_name: Name of function (string).
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

        self.total_weighted_crps = self.add_weight(
            name='total_weighted_crps', initializer='zeros'
        )
        self.total_weight = self.add_weight(
            name='total_weight', initializer='zeros'
        )

    def update_state(self, target_tensor, prediction_tensor,
                     sample_weight=None):
        """Updates CRPS.

        B = batch size (number of data samples)

        :param prediction_tensor: B-by-2 tensor of predicted values, where
            prediction_tensor[:, 0] contains means and prediction_tensor[:, 1]
            contains standard deviations.
        :param target_tensor: length-B tensor of actual values.
        :param sample_weight: Leave this alone.
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
        self.total_weighted_crps.assign_add(keras.ops.mean(crps_tensor_1d))
        self.total_weight.assign_add(1.)

    def result(self):
        """Computes final CRPS.

        :return: crps_value: CRPS (a scalar value).
        """

        return (
            self.total_weighted_crps /
            keras.ops.maximum(self.total_weight, 1e-7)
        )

    def reset_state(self):
        """Resets values between epochs."""

        self.total_weighted_crps.assign(0.)
        self.total_weight.assign(0.)


class QuantileLoss(keras.metrics.Metric):
    def __init__(self, function_name, quantile_levels, **kwargs):
        """Turns quantile loss into metric.

        :param function_name: Name of function (string).
        :param quantile_levels: 1-D numpy array of quantile levels, all ranging
            from (0, 1).
        """

        assert isinstance(function_name, str)
        super().__init__(name=function_name, **kwargs)

        assert len(quantile_levels.shape) == 1
        assert numpy.all(quantile_levels > 0.)
        assert numpy.all(quantile_levels < 1.)

        self.quantile_levels = quantile_levels

        self.total_weighted_quantile_loss = self.add_weight(
            name='total_weighted_quantile_loss', initializer='zeros'
        )
        self.total_weight = self.add_weight(
            name='total_weight', initializer='zeros'
        )

    def update_state(self, target_tensor, prediction_tensor,
                     sample_weight=None):
        """Updates quantile loss.

        B = batch size (number of data samples)
        Q = number of quantiles

        :param target_tensor: length-B tensor of actual values.
        :param prediction_tensor: B-by-Q tensor of predicted quantiles.
        :param sample_weight: Leave this alone.
        """

        quantile_level_tensor = keras.ops.convert_to_tensor(
            self.quantile_levels, dtype=prediction_tensor.dtype
        )
        quantile_level_tensor_2d = keras.ops.reshape(
            quantile_level_tensor, (1, len(self.quantile_levels))
        )

        target_tensor = keras.ops.cast(target_tensor, prediction_tensor.dtype)
        target_tensor_2d = keras.ops.expand_dims(target_tensor, axis=-1)
        target_tensor_2d = keras.ops.repeat(
            target_tensor_2d, len(self.quantile_levels), axis=-1
        )

        error_tensor_2d = target_tensor_2d - prediction_tensor

        quantile_loss_tensor_2d = keras.ops.maximum(
            quantile_level_tensor_2d * error_tensor_2d,
            (quantile_level_tensor_2d - 1.) * error_tensor_2d
        )

        self.total_weighted_quantile_loss.assign_add(
            keras.ops.mean(quantile_loss_tensor_2d)
        )
        self.total_weight.assign_add(1.)

    def result(self):
        """Computes final quantile loss.

        :return: quantile_loss: Quantile loss (a scalar value).
        """

        return (
            self.total_weighted_quantile_loss /
            keras.ops.maximum(self.total_weight, 1e-7)
        )

    def reset_state(self):
        """Resets values between epochs."""

        self.total_weighted_quantile_loss.assign(0.)
        self.total_weight.assign(0.)
