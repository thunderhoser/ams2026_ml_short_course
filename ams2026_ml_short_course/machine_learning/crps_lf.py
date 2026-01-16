"""Helper methods for CRPS loss function (CRPS-LF) approach."""

import copy
import numpy
import keras
from ams2026_ml_short_course.utils import utils
from ams2026_ml_short_course.machine_learning import custom_losses
from ams2026_ml_short_course.machine_learning import custom_metrics

KERNEL_INITIALIZER_NAME = 'glorot_uniform'
BIAS_INITIALIZER_NAME = 'zeros'

DEFAULT_INPUT_DIMENSIONS = numpy.array([32, 32, 4], dtype=int)
DEFAULT_CONV_BLOCK_LAYER_COUNTS = numpy.array([2, 2, 2, 2], dtype=int)
DEFAULT_CONV_CHANNEL_COUNTS = numpy.array(
    [32, 32, 64, 64, 128, 128, 256, 256], dtype=int
)
DEFAULT_CONV_DROPOUT_RATES = numpy.full(8, 0.)
DEFAULT_CONV_FILTER_SIZES = numpy.full(8, 3, dtype=int)
DEFAULT_DENSE_NEURON_COUNTS = numpy.array([776, 147, 100, 100, 100], dtype=int)
DEFAULT_DENSE_DROPOUT_RATES = numpy.array([0.5, 0.5, 0.5, 0.5, 0])
DEFAULT_INNER_ACTIV_FUNCTION_NAME = copy.deepcopy(utils.RELU_FUNCTION_NAME)
DEFAULT_INNER_ACTIV_FUNCTION_ALPHA = 0.2
DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME = copy.deepcopy(utils.RELU_FUNCTION_NAME)
DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA = 0.
DEFAULT_L1_WEIGHT = 0.
DEFAULT_L2_WEIGHT = 0.001


def _get_2d_conv_layer(
        num_rows_in_filter, num_columns_in_filter, num_rows_per_stride,
        num_columns_per_stride, num_filters, use_edge_padding=True,
        weight_regularizer=None):
    """Creates layer for 2-D convolution.

    :param num_rows_in_filter: Number of rows in each filter (kernel).
    :param num_columns_in_filter: Number of columns in each filter (kernel).
    :param num_rows_per_stride: Number of rows per filter stride.
    :param num_columns_per_stride: Number of columns per filter stride.
    :param num_filters: Number of filters (output channels).
    :param use_edge_padding: Boolean flag.  If True, output grid will be same
        size as input grid.  If False, output grid may be smaller.
    :param weight_regularizer: Will be used to regularize weights in the new
        layer.  This may be instance of `keras.regularizers` or None (if you
        want no regularization).
    :return: layer_object: Instance of `keras.layers.Conv2D`.
    """

    return keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=(num_rows_in_filter, num_columns_in_filter),
        strides=(num_rows_per_stride, num_columns_per_stride),
        padding='same' if use_edge_padding else 'valid',
        dilation_rate=(1, 1), activation=None, use_bias=True,
        kernel_initializer=KERNEL_INITIALIZER_NAME,
        bias_initializer=BIAS_INITIALIZER_NAME,
        kernel_regularizer=weight_regularizer,
        bias_regularizer=weight_regularizer
    )


def _get_2d_pooling_layer(
        num_rows_in_window, num_columns_in_window, num_rows_per_stride,
        num_columns_per_stride, do_max_pooling=True):
    """Creates layer for 2-D pooling.

    :param num_rows_in_window: Number of rows in pooling window.
    :param num_columns_in_window: Number of columns in pooling window.
    :param num_rows_per_stride: Number of rows per window stride.
    :param num_columns_per_stride: Number of columns per window stride.
    :param do_max_pooling: Boolean flag.  If True (False), will do max-
        (average-)pooling.
    :return: layer_object: Instance of `keras.layers.MaxPooling2D` or
        `keras.layers.AveragePooling2D`.
    """

    if do_max_pooling:
        return keras.layers.MaxPooling2D(
            pool_size=(num_rows_in_window, num_columns_in_window),
            strides=(num_rows_per_stride, num_columns_per_stride),
            padding='valid'
        )

    return keras.layers.AveragePooling2D(
        pool_size=(num_rows_in_window, num_columns_in_window),
        strides=(num_rows_per_stride, num_columns_per_stride),
        padding='valid'
    )


def setup_cnn(
        loss_function, metric_functions,
        input_dimensions=DEFAULT_INPUT_DIMENSIONS,
        conv_block_layer_counts=DEFAULT_CONV_BLOCK_LAYER_COUNTS,
        conv_layer_channel_counts=DEFAULT_CONV_CHANNEL_COUNTS,
        conv_layer_dropout_rates=DEFAULT_CONV_DROPOUT_RATES,
        conv_layer_filter_sizes=DEFAULT_CONV_FILTER_SIZES,
        dense_layer_neuron_counts=DEFAULT_DENSE_NEURON_COUNTS,
        dense_layer_dropout_rates=DEFAULT_DENSE_DROPOUT_RATES,
        inner_activ_function_name=DEFAULT_INNER_ACTIV_FUNCTION_NAME,
        inner_activ_function_alpha=DEFAULT_INNER_ACTIV_FUNCTION_ALPHA,
        output_activ_function_name=DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME,
        output_activ_function_alpha=DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        use_batch_normalization=True):
    """Sets up (but does not train) CNN with CRPS-LF approach to UQ.

    This method sets up the architecture, loss function, and optimizer.

    B = number of convolutional blocks
    C = number of convolutional layers
    D = number of dense layers

    :param loss_function: Loss function.
    :param metric_functions: 1-D list of metrics.
    :param input_dimensions: numpy array with dimensions of input data.  Entries
        should be (num_grid_rows, num_grid_columns, num_channels).
    :param conv_block_layer_counts: length-B numpy array with number of
        convolutional layers in each block.  Recall that each conv block except
        the last ends with a pooling layer.
    :param conv_layer_channel_counts: length-C numpy array with number of
        channels (filters) produced by each convolutional layer.
    :param conv_layer_dropout_rates: length-C numpy array of dropout rates.  To
        turn off dropout for a given layer, use NaN or a non-positive number.
    :param conv_layer_filter_sizes: length-C numpy array of filter sizes.  All
        filters will be square (num rows = num columns).
    :param dense_layer_neuron_counts: length-D numpy array with number of
        neurons for each dense layer.  The last value in this array is the
        number of target variables (predictands).
    :param dense_layer_dropout_rates: length-D numpy array of dropout rates.  To
        turn off dropout for a given layer, use NaN or a non-positive number.
    :param inner_activ_function_name: Name of activation function for all inner
        (non-output) layers.
    :param inner_activ_function_alpha: Alpha (slope parameter) for
        activation function for all inner layers.  Applies only to ReLU and eLU.
    :param output_activ_function_name: Same as `inner_activ_function_name` but
        for output layer.
    :param output_activ_function_alpha: Same as `inner_activ_function_alpha` but
        for output layer.
    :param l1_weight: Weight for L_1 regularization.
    :param l2_weight: Weight for L_2 regularization.
    :param use_batch_normalization: Boolean flag.  If True, will use batch
        normalization after each inner layer.

    :return: model_object: Untrained instance of `keras.models.Model`.
    """

    num_conv_layers = len(conv_layer_channel_counts)
    assert numpy.sum(conv_block_layer_counts) == num_conv_layers

    num_input_rows = input_dimensions[0]
    num_input_columns = input_dimensions[1]
    num_input_channels = input_dimensions[2]

    input_layer_object = keras.layers.Input(
        shape=(num_input_rows, num_input_columns, num_input_channels)
    )
    regularizer_object = utils._get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    layer_object = None

    for i in range(num_conv_layers):
        if layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = layer_object

        layer_object = _get_2d_conv_layer(
            num_rows_in_filter=conv_layer_filter_sizes[i],
            num_columns_in_filter=conv_layer_filter_sizes[i],
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=conv_layer_channel_counts[i], use_edge_padding=True,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        layer_object = utils._get_activation_layer(
            function_name=inner_activ_function_name,
            slope_param=inner_activ_function_alpha
        )(layer_object)

        if conv_layer_dropout_rates[i] > 0:
            layer_object = utils._get_dropout_layer(
                dropout_fraction=conv_layer_dropout_rates[i]
            )(layer_object)

        if use_batch_normalization:
            layer_object = utils._get_batch_norm_layer()(layer_object)

        if i + 1 not in numpy.cumsum(conv_block_layer_counts):
            continue

        if i == num_conv_layers - 1:
            continue

        layer_object = _get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2, do_max_pooling=True
        )(layer_object)

    layer_object = keras.layers.Flatten()(layer_object)

    num_dense_layers = len(dense_layer_neuron_counts)

    for i in range(num_dense_layers):
        if layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = layer_object

        layer_object = utils._get_dense_layer(
            num_output_units=dense_layer_neuron_counts[i],
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        if i == num_dense_layers - 1:
            layer_object = utils._get_activation_layer(
                function_name=output_activ_function_name,
                slope_param=output_activ_function_alpha
            )(layer_object)
        else:
            layer_object = utils._get_activation_layer(
                function_name=inner_activ_function_name,
                slope_param=inner_activ_function_alpha
            )(layer_object)

        if dense_layer_dropout_rates[i] > 0:
            layer_object = utils._get_dropout_layer(
                dropout_fraction=dense_layer_dropout_rates[i]
            )(layer_object)

        if use_batch_normalization and i != num_dense_layers - 1:
            layer_object = utils._get_batch_norm_layer()(layer_object)

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object
    )

    model_object.compile(
        loss=loss_function,
        optimizer=keras.optimizers.Adam(),
        metrics=metric_functions
    )

    model_object.summary()
    return model_object
