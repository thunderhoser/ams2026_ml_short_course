"""Helper methods for running discard test."""

import numpy
from matplotlib import pyplot
from ams2026_ml_short_course.utils import spread_skill_utils

DISCARD_FRACTIONS_KEY = 'discard_fractions'
ERROR_VALUES_KEY = 'error_values'
EXAMPLE_FRACTIONS_KEY = 'example_fractions'
MEAN_MEAN_PREDICTIONS_KEY = 'mean_mean_predictions'
MEAN_TARGET_VALUES_KEY = 'mean_target_values'
MONOTONICITY_FRACTION_KEY = 'monotonicity_fraction'
DISCARD_IMPROVEMENT_KEY = 'discard_improvement'

DEFAULT_LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
DEFAULT_LINE_WIDTH = 3.

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

INSET_FONT_SIZE = 14
DEFAULT_FONT_SIZE = 24
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)


def get_stdev_uncertainty_function():
    """Creates function to compute stdev of predictive distribution.

    :return: uncertainty_function: Function handle.
    """

    def uncertainty_function(target_values, prediction_matrix):
        """Computes stdev of predictive distribution for each example.

        E = number of examples
        S = number of ensemble members

        :param target_values: length-E numpy array of actual values.
        :param prediction_matrix: E-by-S numpy array of predicted values.
        :return: stdev_by_example: length-E numpy array with standard
            deviations.
        """

        return numpy.mean(prediction_matrix, ddof=1, axis=-1)

    return uncertainty_function


def get_rmse_error_function():
    """Creates function to compute RMSE of ensemble mean.

    :return: error_function: Function handle.
    """

    def error_function(target_values, prediction_matrix, use_example_flags):
        """Computes RMSE of ensemble mean for each example.

        E = number of examples
        S = number of ensemble members

        :param target_values: length-E numpy array of actual values.
        :param prediction_matrix: E-by-S numpy array of predicted values.
        :param use_example_flags: length-E numpy array of Boolean flags,
            indicating which examples to include.
        :return: rmse_value: RMSE (scalar).
        """

        squared_errors = (
            target_values[use_example_flags] -
            numpy.mean(prediction_matrix[use_example_flags, :], axis=-1)
        ) ** 2

        numpy.sqrt(numpy.mean(squared_errors))

    return error_function


def run_discard_test(
        target_values, prediction_matrix, discard_fractions, error_function,
        uncertainty_function):
    """Runs the discard test.

    E = number of examples
    S = number of ensemble members
    F = number of discard fractions

    :param target_values: length-E numpy array of actual values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :param discard_fractions: length-(F - 1) numpy array of discard fractions,
        ranging from (0, 1).  This method will use 0 as the lowest discard
        fraction.

    :param error_function: Function with the following inputs and outputs...
    Input: target_values: See above.
    Input: prediction_matrix: See above.
    Input: use_example_flags: length-E numpy array of Boolean flags, indicating
        which data examples to use.
    Output: error_value: Scalar value of error metric.  The metric must be
        oriented so that higher values = worse error.

    :param uncertainty_function: Function with the following inputs and
        outputs...
    Input: target_values: See above.
    Input: prediction_matrix: See above.
    Output: uncertainty_values: length-E numpy array with values of uncertainty
        metric.  The metric must be oriented so that higher values
        = more uncertainty.

    :return: result_dict: Dictionary with the following keys.
    result_dict['discard_fractions']: length-F numpy array of discard fractions,
        sorted in increasing order.
    result_dict['error_values']: length-F numpy array of corresponding error
        values.
    result_dict['example_fractions']: length-F numpy array with fraction of
        examples left after each discard.
    result_dict['mean_mean_predictions']: length-F numpy array, where the
        [i]th entry is the mean central (mean ensemble-mean) prediction for the
        [i]th discard fraction.
    result_dict['mean_target_values']: length-F numpy array, where the [i]th
        entry is the mean target value for the [i]th discard fraction.
    result_dict['monotonicity_fraction']: Monotonicity fraction.  This is the
        fraction of times that the error function improves when discard
        fraction is increased.
    result_dict['discard_improvement']: Discard improvement.  This is the mean
        improvement in the error metric per 1% of data discarded.
    """

    # Check input args.
    assert not numpy.any(numpy.isnan(target_values))
    assert len(target_values.shape) == 1

    assert not numpy.any(numpy.isnan(prediction_matrix))
    assert len(prediction_matrix.shape) == 2

    num_examples = len(target_values)
    num_ensemble_members = prediction_matrix.shape[1]
    assert num_ensemble_members > 1

    these_dim = numpy.array([num_examples, num_ensemble_members], dtype=int)
    assert numpy.array_equal(
        these_dim,
        numpy.array(prediction_matrix.shape, dtype=int)
    )

    assert len(discard_fractions.shape) == 1
    assert numpy.all(discard_fractions > 0.)
    assert numpy.all(numpy.diff(discard_fractions) > 0.)

    discard_fractions = numpy.concatenate((
        numpy.array([0.]),
        discard_fractions
    ))

    num_fractions = len(discard_fractions)
    assert num_fractions >= 2

    # Do actual stuff.
    uncertainty_values = uncertainty_function(target_values, prediction_matrix)
    mean_predictions = numpy.mean(prediction_matrix, axis=-1)

    error_values = numpy.full(num_fractions, numpy.nan)
    example_fractions = numpy.full(num_fractions, numpy.nan)
    mean_mean_predictions = numpy.full(num_fractions, numpy.nan)
    mean_target_values = numpy.full(num_fractions, numpy.nan)
    use_example_flags = numpy.full(uncertainty_values.shape, 1, dtype=bool)

    for k in range(num_fractions):
        this_percentile_level = 100 * (1 - discard_fractions[k])
        this_inverted_mask = (
            uncertainty_values >
            numpy.percentile(uncertainty_values, this_percentile_level)
        )
        use_example_flags[this_inverted_mask] = False

        example_fractions[k] = numpy.mean(use_example_flags)
        error_values[k] = error_function(
            target_values, prediction_matrix, use_example_flags
        )
        mean_mean_predictions[k] = numpy.mean(
            mean_predictions[use_example_flags]
        )
        mean_target_values[k] = numpy.mean(
            target_values[use_example_flags]
        )

    monotonicity_fraction = numpy.mean(numpy.diff(error_values) < 0)
    discard_improvement = -1 * numpy.mean(
        numpy.diff(error_values) / numpy.diff(discard_fractions)
    )

    return {
        DISCARD_FRACTIONS_KEY: discard_fractions,
        ERROR_VALUES_KEY: error_values,
        EXAMPLE_FRACTIONS_KEY: example_fractions,
        MEAN_MEAN_PREDICTIONS_KEY: mean_mean_predictions,
        MEAN_TARGET_VALUES_KEY: mean_target_values,
        MONOTONICITY_FRACTION_KEY: monotonicity_fraction,
        DISCARD_IMPROVEMENT_KEY: discard_improvement
    }


def plot_discard_test(
        result_dict, model_description_string,
        line_colour=DEFAULT_LINE_COLOUR,
        line_style='solid', line_width=DEFAULT_LINE_WIDTH):
    """Plots results of discard test.

    :param result_dict: Dictionary returned by `run_discard_test`.
    :param model_description_string: Model or dataset description (will go at
        beginning of title).
    :param line_colour: Line colour (in any format accepted by matplotlib).
    :param line_style: Line style (in any format accepted by matplotlib).
    :param line_width: Line width (in any format accepted by matplotlib).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    discard_fractions = result_dict[DISCARD_FRACTIONS_KEY]
    error_values = result_dict[ERROR_VALUES_KEY]
    # example_fractions = result_dict[EXAMPLE_FRACTIONS_KEY]
    mean_mean_predictions = result_dict[MEAN_MEAN_PREDICTIONS_KEY]
    mean_target_values = result_dict[MEAN_TARGET_VALUES_KEY]
    monotonicity_fraction = result_dict[MONOTONICITY_FRACTION_KEY]
    discard_improvement = result_dict[DISCARD_IMPROVEMENT_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.plot(
        discard_fractions, error_values,
        color=line_colour, linestyle=line_style, linewidth=line_width,
        marker='o', markersize=12, markeredgewidth=0,
        markerfacecolor=line_colour, markeredgecolor=line_colour
    )

    axes_object.set_xlabel('Discard fraction')
    axes_object.set_ylabel('Error in ensemble mean')
    axes_object.set_xlim(left=0.)

    inset_axes_object = spread_skill_utils._plot_means_as_inset(
        figure_object=figure_object,
        bin_centers=discard_fractions,
        bin_mean_predictions=mean_mean_predictions,
        bin_mean_target_values=mean_target_values,
        plotting_corner_string='top_right',
        for_spread_skill_plot=False
    )
    inset_axes_object.set_zorder(axes_object.get_zorder() + 1)

    inset_axes_object.set_xticks(axes_object.get_xticks())
    inset_axes_object.set_xlim(axes_object.get_xlim())
    # inset_axes_object.set_xlabel(
    #     'Discard fraction',
    #     fontsize=INSET_FONT_SIZE
    # )
    unit_string = r's$^{-1}$'
    inset_axes_object.set_ylabel(
        'Avg target or pred ({0:s})'.format(unit_string),
        fontsize=INSET_FONT_SIZE
    )
    inset_axes_object.set_title(
        'Avgs by discard fraction', fontsize=INSET_FONT_SIZE
    )

    title_string = (
        '{0:s}\n'
        'Discard test for max future vorticity\n'
        'MF = {1:.4f}; DI = {2:.6f}'
    ).format(
        model_description_string,
        monotonicity_fraction,
        discard_improvement
    )

    axes_object.set_title(title_string)
    return figure_object, axes_object
