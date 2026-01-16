"""Helper methods."""

import copy
import errno
import os.path
import time
import calendar
import numpy
import matplotlib.colors
from matplotlib import pyplot
import keras
import keras.layers as layers
from ams2026_ml_short_course.plotting import evaluation_plotting

# Variable names.
METADATA_COLUMNS_ORIG = [
    'Step_ID', 'Track_ID', 'Ensemble_Name', 'Ensemble_Member', 'Run_Date',
    'Valid_Date', 'Forecast_Hour', 'Valid_Hour_UTC'
]

EXTRANEOUS_COLUMNS_ORIG = [
    'Duration', 'Centroid_Lon', 'Centroid_Lat', 'Centroid_X', 'Centroid_Y',
    'Storm_Motion_U', 'Storm_Motion_V', 'Matched', 'Max_Hail_Size',
    'Num_Matches', 'Shape', 'Location', 'Scale'
]

TARGET_NAME_ORIG = 'RVORT1_MAX-future_max'
TARGET_NAME = 'max_future_vorticity_s01'
BINARIZED_TARGET_NAME = 'strong_future_rotation_flag'
AREA_NAME = 'area_km2'
MAJOR_AXIS_NAME = 'major_axis_km'
MINOR_AXIS_NAME = 'minor_axis_km'
ORIENTATION_NAME = 'orientation_deg'

METADATA_COLUMNS_ORIG_TO_NEW = {
    'Step_ID': 'storm_object_name',
    'Track_ID': 'storm_cell_name',
    'Ensemble_Name': 'ensemble_name',
    'Ensemble_Member': 'ensemble_member_name',
    'Run_Date': 'init_time_string',
    'Valid_Date': 'valid_time_string',
    'Forecast_Hour': 'lead_time_hours',
    'Valid_Hour_UTC': 'valid_hour'
}

TARGET_COLUMNS_ORIG_TO_NEW = {
    TARGET_NAME_ORIG: TARGET_NAME
}

PREDICTOR_COLUMNS_ORIG_TO_NEW = {
    'REFL_COM_mean': 'composite_refl_mean_dbz',
    'REFL_COM_max': 'composite_refl_max_dbz',
    'REFL_COM_min': 'composite_refl_min_dbz',
    'REFL_COM_std': 'composite_refl_stdev_dbz',
    'REFL_COM_percentile_10': 'composite_refl_prctile10_dbz',
    'REFL_COM_percentile_25': 'composite_refl_prctile25_dbz',
    'REFL_COM_percentile_50': 'composite_refl_median_dbz',
    'REFL_COM_percentile_75': 'composite_refl_prctile75_dbz',
    'REFL_COM_percentile_90': 'composite_refl_prctile90_dbz',
    'U10_mean': 'u_wind_10metres_mean_m_s01',
    'U10_max': 'u_wind_10metres_max_m_s01',
    'U10_min': 'u_wind_10metres_min_m_s01',
    'U10_std': 'u_wind_10metres_stdev_m_s01',
    'U10_percentile_10': 'u_wind_10metres_prctile10_m_s01',
    'U10_percentile_25': 'u_wind_10metres_prctile25_m_s01',
    'U10_percentile_50': 'u_wind_10metres_median_m_s01',
    'U10_percentile_75': 'u_wind_10metres_prctile75_m_s01',
    'U10_percentile_90': 'u_wind_10metres_prctile90_m_s01',
    'V10_mean': 'v_wind_10metres_mean_m_s01',
    'V10_max': 'v_wind_10metres_max_m_s01',
    'V10_min': 'v_wind_10metres_min_m_s01',
    'V10_std': 'v_wind_10metres_stdev_m_s01',
    'V10_percentile_10': 'v_wind_10metres_prctile10_m_s01',
    'V10_percentile_25': 'v_wind_10metres_prctile25_m_s01',
    'V10_percentile_50': 'v_wind_10metres_median_m_s01',
    'V10_percentile_75': 'v_wind_10metres_prctile75_m_s01',
    'V10_percentile_90': 'v_wind_10metres_prctile90_m_s01',
    'T2_mean': 'temperature_2metres_mean_kelvins',
    'T2_max': 'temperature_2metres_max_kelvins',
    'T2_min': 'temperature_2metres_min_kelvins',
    'T2_std': 'temperature_2metres_stdev_kelvins',
    'T2_percentile_10': 'temperature_2metres_prctile10_kelvins',
    'T2_percentile_25': 'temperature_2metres_prctile25_kelvins',
    'T2_percentile_50': 'temperature_2metres_median_kelvins',
    'T2_percentile_75': 'temperature_2metres_prctile75_kelvins',
    'T2_percentile_90': 'temperature_2metres_prctile90_kelvins',
    'area': AREA_NAME,
    'eccentricity': 'eccentricity',
    'major_axis_length': MAJOR_AXIS_NAME,
    'minor_axis_length': MINOR_AXIS_NAME,
    'orientation': ORIENTATION_NAME
}

MAE_KEY = 'mean_absolute_error'
RMSE_KEY = 'root_mean_squared_error'
MEAN_BIAS_KEY = 'mean_bias'
MAE_SKILL_SCORE_KEY = 'mae_skill_score'
MSE_SKILL_SCORE_KEY = 'mse_skill_score'

MAX_PEIRCE_SCORE_KEY = 'max_peirce_score'
AUC_KEY = 'area_under_roc_curve'
MAX_CSI_KEY = 'max_csi'
BRIER_SCORE_KEY = 'brier_score'
BRIER_SKILL_SCORE_KEY = 'brier_skill_score'

PREDICTORS_KEY = 'predictor_matrix'
PERMUTED_FLAGS_KEY = 'permuted_flags'
PERMUTED_INDICES_KEY = 'permuted_predictor_indices'
PERMUTED_COSTS_KEY = 'permuted_cost_matrix'
DEPERMUTED_INDICES_KEY = 'depermuted_predictor_indices'
DEPERMUTED_COSTS_KEY = 'depermuted_cost_matrix'

HIT_INDICES_KEY = 'hit_indices'
MISS_INDICES_KEY = 'miss_indices'
FALSE_ALARM_INDICES_KEY = 'false_alarm_indices'
CORRECT_NULL_INDICES_KEY = 'correct_null_indices'

# Plotting constants.
FIGURE_WIDTH_INCHES = 10
FIGURE_HEIGHT_INCHES = 10
LARGE_FIGURE_WIDTH_INCHES = 15
LARGE_FIGURE_HEIGHT_INCHES = 15

DEFAULT_GRAPH_LINE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
DEFAULT_GRAPH_LINE_WIDTH = 2

BAR_GRAPH_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
BAR_GRAPH_EDGE_WIDTH = 2
BAR_GRAPH_FONT_SIZE = 14
BAR_GRAPH_FONT_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

GREEN_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
ORANGE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
PURPLE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
GREY_COLOUR = numpy.full(3, 152. / 255)

FONT_SIZE = 20
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

# Misc constants.
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'

GRID_SPACING_KM = 3.
RADIANS_TO_DEGREES = 180. / numpy.pi

RANDOM_SEED = 6695
LAMBDA_TOLERANCE = 1e-10

ELU_FUNCTION_NAME = 'elu'
RELU_FUNCTION_NAME = 'relu'
SELU_FUNCTION_NAME = 'selu'
TANH_FUNCTION_NAME = 'tanh'
SIGMOID_FUNCTION_NAME = 'sigmoid'

ACTIVATION_FUNCTION_NAMES = [
    ELU_FUNCTION_NAME, RELU_FUNCTION_NAME, SELU_FUNCTION_NAME,
    TANH_FUNCTION_NAME, SIGMOID_FUNCTION_NAME
]

KERNEL_INITIALIZER_NAME = 'glorot_uniform'
BIAS_INITIALIZER_NAME = 'zeros'

DEFAULT_NEURON_COUNTS = numpy.array([1000, 178, 32, 6, 1], dtype=int)
DEFAULT_DROPOUT_RATES = numpy.array([0.5, 0.5, 0.5, 0.5, 0])
DEFAULT_INNER_ACTIV_FUNCTION_NAME = copy.deepcopy(RELU_FUNCTION_NAME)
DEFAULT_INNER_ACTIV_FUNCTION_ALPHA = 0.2
DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME = copy.deepcopy(SIGMOID_FUNCTION_NAME)
DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA = 0.
DEFAULT_L1_WEIGHT = 0.
DEFAULT_L2_WEIGHT = 0.001

PLATEAU_PATIENCE_EPOCHS = 5
PLATEAU_LEARNING_RATE_MULTIPLIER = 0.6
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 10
LOSS_PATIENCE = 0.

DEFAULT_NUM_BOOTSTRAP_REPS = 1000

ORIGINAL_COST_KEY = 'orig_cost_estimates'
BEST_PREDICTORS_KEY = 'best_predictor_names'
BEST_COSTS_KEY = 'best_cost_matrix'
STEP1_PREDICTORS_KEY = 'step1_predictor_names'
STEP1_COSTS_KEY = 'step1_cost_matrix'
BACKWARDS_FLAG_KEY = 'is_backwards_test'



def _get_reliability_curve(actual_values, predicted_values, num_bins,
                           max_bin_edge, invert=False):
    """Computes reliability curve for one target variable.

    E = number of examples
    B = number of bins

    :param actual_values: length-E numpy array of actual values.
    :param predicted_values: length-E numpy array of predicted values.
    :param num_bins: Number of bins (points in curve).
    :param max_bin_edge: Value at upper edge of last bin.
    :param invert: Boolean flag.  If True, will return inverted reliability
        curve, which bins by target value and relates target value to
        conditional mean prediction.  If False, will return normal reliability
        curve, which bins by predicted value and relates predicted value to
        conditional mean observation (target).
    :return: mean_predictions: length-B numpy array of x-coordinates.
    :return: mean_observations: length-B numpy array of y-coordinates.
    :return: example_counts: length-B numpy array with num examples in each bin.
    """

    max_bin_edge = max([max_bin_edge, numpy.finfo(float).eps])
    bin_cutoffs = numpy.linspace(0., max_bin_edge, num=num_bins + 1)

    bin_index_by_example = numpy.digitize(
        actual_values if invert else predicted_values, bin_cutoffs, right=False
    ) - 1
    bin_index_by_example[bin_index_by_example < 0] = 0
    bin_index_by_example[bin_index_by_example > num_bins - 1] = num_bins - 1

    mean_predictions = numpy.full(num_bins, numpy.nan)
    mean_observations = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, -1, dtype=int)

    for i in range(num_bins):
        these_example_indices = numpy.where(bin_index_by_example == i)[0]

        example_counts[i] = len(these_example_indices)
        mean_predictions[i] = numpy.mean(
            predicted_values[these_example_indices]
        )
        mean_observations[i] = numpy.mean(actual_values[these_example_indices])

    return mean_predictions, mean_observations, example_counts


def _add_colour_bar(
        axes_object, colour_map_object, values_to_colour, min_colour_value,
        max_colour_value, colour_norm_object=None,
        orientation_string='vertical', extend_min=True, extend_max=True):
    """Adds colour bar to existing axes.

    :param axes_object: Existing axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param values_to_colour: numpy array of values to colour.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.  If `colour_norm_object is None`,
        will assume that scale is linear.
    :param orientation_string: Orientation of colour bar ("vertical" or
        "horizontal").
    :param extend_min: Boolean flag.  If True, the bottom of the colour bar will
        have an arrow.  If False, it will be a flat line, suggesting that lower
        values are not possible.
    :param extend_max: Same but for top of colour bar.
    :return: colour_bar_object: Colour bar (instance of
        `matplotlib.pyplot.colorbar`) created by this method.
    """

    if colour_norm_object is None:
        colour_norm_object = matplotlib.colors.Normalize(
            vmin=min_colour_value, vmax=max_colour_value, clip=False
        )

    scalar_mappable_object = pyplot.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object
    )
    scalar_mappable_object.set_array(values_to_colour)

    if extend_min and extend_max:
        extend_string = 'both'
    elif extend_min:
        extend_string = 'min'
    elif extend_max:
        extend_string = 'max'
    else:
        extend_string = 'neither'

    if orientation_string == 'horizontal':
        padding = 0.075
    else:
        padding = 0.05

    colour_bar_object = pyplot.colorbar(
        ax=axes_object, mappable=scalar_mappable_object,
        orientation=orientation_string, pad=padding, extend=extend_string,
        shrink=0.8
    )

    colour_bar_object.ax.tick_params(labelsize=FONT_SIZE)
    return colour_bar_object


def _get_weight_regularizer(l1_weight, l2_weight):
    """Creates regularizer for neural-net weights.

    :param l1_weight: L1 regularization weight.  This "weight" is not to be
        confused with those being regularized (weights learned by the net).
    :param l2_weight: L2 regularization weight.
    :return: regularizer_object: Instance of `keras.regularizers.l1_l2`.
    """

    l1_weight = numpy.nanmax(numpy.array([l1_weight, 0.]))
    l2_weight = numpy.nanmax(numpy.array([l2_weight, 0.]))

    return keras.regularizers.l1_l2(l1=l1_weight, l2=l2_weight)


def _get_dense_layer(num_output_units, weight_regularizer=None):
    """Creates dense (fully connected) layer.

    :param num_output_units: Number of output units (or "features" or
        "neurons").
    :param weight_regularizer: Will be used to regularize weights in the new
        layer.  This may be instance of `keras.regularizers` or None (if you
        want no regularization).
    :return: layer_object: Instance of `keras.layers.Dense`.
    """

    return keras.layers.Dense(
        num_output_units, activation=None, use_bias=True,
        kernel_initializer=KERNEL_INITIALIZER_NAME,
        bias_initializer=BIAS_INITIALIZER_NAME,
        kernel_regularizer=weight_regularizer,
        bias_regularizer=weight_regularizer
    )


def _get_activation_layer(function_name, slope_param=0.2):
    """Creates activation layer.

    :param function_name: Name of activation function.
    :param slope_param: Slope parameter (alpha) for activation function.  Used
        only for eLU and ReLU.
    :return: layer_object: Instance of `keras.layers.Activation`,
        `keras.layers.ELU`, or `keras.layers.LeakyReLU`.
    """

    assert function_name in ACTIVATION_FUNCTION_NAMES

    if function_name == ELU_FUNCTION_NAME:
        return keras.layers.ELU(alpha=slope_param)

    if function_name == RELU_FUNCTION_NAME:
        if slope_param <= 0:
            return keras.layers.ReLU()

        return keras.layers.LeakyReLU(alpha=slope_param)

    return keras.layers.Activation(function_name)


def _get_dropout_layer(dropout_fraction):
    """Creates dropout layer.

    :param dropout_fraction: Fraction of weights to drop.
    :return: layer_object: Instance of `keras.layers.Dropout`.
    """

    assert dropout_fraction > 0.
    assert dropout_fraction < 1.

    return keras.layers.Dropout(rate=dropout_fraction)


def _get_batch_norm_layer():
    """Creates batch-normalization layer.

    :return: Instance of `keras.layers.BatchNormalization`.
    """

    return keras.layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True
    )


def _mkdir_recursive_if_necessary(directory_name=None, file_name=None):
    """Creates directory if necessary (i.e., doesn't already exist).

    This method checks for the argument `directory_name` first.  If
    `directory_name` is None, this method checks for `file_name` and extracts
    the directory.

    :param directory_name: Path to local directory.
    :param file_name: Path to local file.
    """

    if directory_name is None:
        directory_name = os.path.dirname(file_name)
    if directory_name == '':
        return

    try:
        os.makedirs(directory_name)
    except OSError as this_error:
        if this_error.errno == errno.EEXIST and os.path.isdir(directory_name):
            pass
        else:
            raise


def create_paneled_figure(
        num_rows, num_columns, figure_width_inches=FIGURE_WIDTH_INCHES,
        figure_height_inches=FIGURE_HEIGHT_INCHES,
        horizontal_spacing=0.075, vertical_spacing=0., shared_x_axis=False,
        shared_y_axis=False, keep_aspect_ratio=True):
    """Creates paneled figure.

    This method only initializes the panels.  It does not plot anything.

    J = number of panel rows
    K = number of panel columns

    :param num_rows: J in the above discussion.
    :param num_columns: K in the above discussion.
    :param figure_width_inches: Width of the entire figure (including all
        panels).
    :param figure_height_inches: Height of the entire figure (including all
        panels).
    :param horizontal_spacing: Spacing (in figure-relative coordinates, from
        0...1) between adjacent panel columns.
    :param vertical_spacing: Spacing (in figure-relative coordinates, from
        0...1) between adjacent panel rows.
    :param shared_x_axis: Boolean flag.  If True, all panels will share the same
        x-axis.
    :param shared_y_axis: Boolean flag.  If True, all panels will share the same
        y-axis.
    :param keep_aspect_ratio: Boolean flag.  If True, the aspect ratio of each
        panel will be preserved (reflect the aspect ratio of the data plotted
        therein).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object_matrix: J-by-K numpy array of axes handles (instances
        of `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object_matrix = pyplot.subplots(
        num_rows, num_columns, sharex=shared_x_axis, sharey=shared_y_axis,
        figsize=(figure_width_inches, figure_height_inches)
    )

    if num_rows == num_columns == 1:
        axes_object_matrix = numpy.full(
            (1, 1), axes_object_matrix, dtype=object
        )

    if num_rows == 1 or num_columns == 1:
        axes_object_matrix = numpy.reshape(
            axes_object_matrix, (num_rows, num_columns)
        )

    pyplot.subplots_adjust(
        left=0.02, bottom=0.02, right=0.98, top=0.95,
        hspace=horizontal_spacing, wspace=vertical_spacing
    )

    if not keep_aspect_ratio:
        return figure_object, axes_object_matrix

    for i in range(num_rows):
        for j in range(num_columns):
            axes_object_matrix[i][j].set(aspect='equal')

    return figure_object, axes_object_matrix


def time_string_to_unix(time_string, time_format):
    """Converts time from string to Unix format.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param time_string: Time string.
    :param time_format: Format of time string (example: "%Y%m%d" or
        "%Y-%m-%d-%H%M%S").
    :return: unix_time_sec: Time in Unix format.
    """

    return calendar.timegm(time.strptime(time_string, time_format))


def time_unix_to_string(unix_time_sec, time_format):
    """Converts time from Unix format to string.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param unix_time_sec: Time in Unix format.
    :param time_format: Desired format of time string (example: "%Y%m%d" or
        "%Y-%m-%d-%H%M%S").
    :return: time_string: Time string.
    """

    return time.strftime(time_format, time.gmtime(unix_time_sec))


def evaluate_regression(
        actual_values, predicted_values, mean_training_target_value,
        verbose=True, create_plots=True, dataset_name=None):
    """Evaluates regression model.

    E = number of examples

    :param actual_values: length-E numpy array of actual target values.
    :param predicted_values: length-E numpy array of predictions.
    :param mean_training_target_value: Mean target value in training data.
    :param verbose: Boolean flag.  If True, will print results to command
        window.
    :param create_plots: Boolean flag.  If True, will create plots.
    :param dataset_name: Dataset name (e.g., "validation").  Used only if
        `create_plots == True or verbose == True`.
    :return: evaluation_dict: Dictionary with the following keys.
    evaluation_dict['mean_absolute_error']: Mean absolute error (MAE).
    evaluation_dict['rmse']: Root mean squared error (RMSE).
    evaluation_dict['mean_bias']: Mean bias (signed error).
    evaluation_dict['mae_skill_score']: MAE skill score (fractional improvement
        over climatology, in range -1...1).
    evaluation_dict['mse_skill_score']: MSE skill score (fractional improvement
        over climatology, in range -1...1).
    """

    signed_errors = predicted_values - actual_values
    mean_bias = numpy.mean(signed_errors)
    mean_absolute_error = numpy.mean(numpy.absolute(signed_errors))
    rmse = numpy.sqrt(numpy.mean(signed_errors ** 2))

    climo_signed_errors = mean_training_target_value - actual_values
    climo_mae = numpy.mean(numpy.absolute(climo_signed_errors))
    climo_mse = numpy.mean(climo_signed_errors ** 2)

    mae_skill_score = (climo_mae - mean_absolute_error) / climo_mae
    mse_skill_score = (climo_mse - rmse ** 2) / climo_mse

    evaluation_dict = {
        MAE_KEY: mean_absolute_error,
        RMSE_KEY: rmse,
        MEAN_BIAS_KEY: mean_bias,
        MAE_SKILL_SCORE_KEY: mae_skill_score,
        MSE_SKILL_SCORE_KEY: mse_skill_score
    }

    if verbose or create_plots:
        dataset_name = dataset_name[0].upper() + dataset_name[1:]

    if verbose:
        print('{0:s} MAE (mean absolute error) = {1:.3e} ks^-1'.format(
            dataset_name, evaluation_dict[MAE_KEY]
        ))
        print('{0:s} MSE (mean squared error) = {1:.3e} ks^-2'.format(
            dataset_name, evaluation_dict[RMSE_KEY]
        ))
        print('{0:s} bias (mean signed error) = {1:.3e} ks^-1'.format(
            dataset_name, evaluation_dict[MEAN_BIAS_KEY]
        ))

        message_string = (
            '{0:s} MAE skill score (improvement over climatology) = {1:.3f}'
        ).format(dataset_name, evaluation_dict[MAE_SKILL_SCORE_KEY])
        print(message_string)

        message_string = (
            '{0:s} MSE skill score (improvement over climatology) = {1:.3f}'
        ).format(dataset_name, evaluation_dict[MSE_SKILL_SCORE_KEY])
        print(message_string)

    if not create_plots:
        return evaluation_dict

    mean_predictions, mean_observations, example_counts = (
        _get_reliability_curve(
            actual_values=actual_values, predicted_values=predicted_values,
            num_bins=20, max_bin_edge=numpy.percentile(predicted_values, 99),
            invert=False
        )
    )

    inv_mean_observations, inv_example_counts = (
        _get_reliability_curve(
            actual_values=actual_values, predicted_values=predicted_values,
            num_bins=20, max_bin_edge=numpy.percentile(actual_values, 99),
            invert=True
        )[1:]
    )

    concat_values = numpy.concatenate((mean_predictions, mean_observations))

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    evaluation_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_predictions=mean_predictions, mean_observations=mean_observations,
        example_counts=example_counts,
        inv_mean_observations=inv_mean_observations,
        inv_example_counts=inv_example_counts,
        mean_value_in_training=mean_training_target_value,
        min_value_to_plot=0., max_value_to_plot=numpy.max(concat_values)
    )

    axes_object.set_xlabel(r'Forecast value (ks$^{-1}$)')
    axes_object.set_ylabel(r'Conditional mean observation (ks$^{-1}$)')

    title_string = '{0:s} attributes diagram for max future vorticity'.format(
        dataset_name
    )
    axes_object.set_title(title_string)
    pyplot.show()

    return evaluation_dict
