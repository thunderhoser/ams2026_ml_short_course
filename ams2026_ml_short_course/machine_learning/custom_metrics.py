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


class MeanSquaredErrorKs02(keras.metrics.Metric):
    def __init__(self, function_name, **kwargs):
        """Same as `MeanSquaredError`, but units are kiloseconds^-2.

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
        target_tensor_2d = 1000 * keras.ops.expand_dims(target_tensor, axis=-1)
        prediction_tensor_2d = 1000 * prediction_tensor

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


class CRPS(keras.metrics.Metric):
    def __init__(self, function_name, **kwargs):
        """Turns CRPS into metric.

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
            keras.ops.sqrt(2. * keras.ops.pi)
        )

        first_term_tensor = standardized_error_tensor * (2. * normal_cdf_tensor - 1.)
        second_term_tensor = 2. * normal_pdf_tensor
        third_term_tensor = -1. / keras.ops.sqrt(keras.ops.pi)

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
