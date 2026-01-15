"""Helper methods for computing spread-skill relationship."""

import numpy
from matplotlib import pyplot

MEAN_PREDICTION_STDEVS_KEY = 'mean_prediction_stdevs'
BIN_EDGE_PREDICTION_STDEVS_KEY = 'bin_edge_prediction_stdevs'
RMSE_VALUES_KEY = 'rmse_values'
SPREAD_SKILL_RELIABILITY_KEY = 'spread_skill_reliability'
SPREAD_SKILL_RATIO_KEY = 'spread_skill_ratio'
EXAMPLE_COUNTS_KEY = 'example_counts'
MEAN_MEAN_PREDICTIONS_KEY = 'mean_mean_predictions'
MEAN_TARGET_VALUES_KEY = 'mean_target_values'

DEFAULT_LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
DEFAULT_LINE_WIDTH = 3.

MEAN_PREDICTION_LINE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
MEAN_PREDICTION_COLOUR_STRING = 'purple'
MEAN_TARGET_LINE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
MEAN_TARGET_COLOUR_STRING = 'green'

INSET_FONT_SIZE = 20

INSET_HISTO_FACE_COLOUR = numpy.full(3, 152. / 255)
INSET_HISTO_EDGE_COLOUR = numpy.full(3, 0.)
INSET_HISTO_EDGE_WIDTH = 1.

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
REFERENCE_LINE_WIDTH = 2.

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15


def _plot_means_as_inset(
        figure_object, bin_centers, bin_mean_predictions,
        bin_mean_target_values, plotting_corner_string, for_spread_skill_plot):
    """Plots means (mean prediction and target by bin) as inset in another fig.

    B = number of bins

    :param figure_object: Will plot as inset in this figure (instance of
        `matplotlib.figure.Figure`).
    :param bin_centers: length-B numpy array with value at center of each bin.
        These values will be plotted on the x-axis.
    :param bin_mean_predictions: length-B numpy array with mean prediction in
        each bin.  These values will be plotted on the y-axis.
    :param bin_mean_target_values: length-B numpy array with mean target value
        (event frequency) in each bin.  These values will be plotted on the
        y-axis.
    :param plotting_corner_string: String in
        ['top_right', 'top_left', 'bottom_right', 'bottom_left'].
    :param for_spread_skill_plot: Boolean flag.
    :return: inset_axes_object: Axes handle for histogram (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    if plotting_corner_string == 'top_right':
        inset_axes_object = figure_object.add_axes([0.625, 0.55, 0.25, 0.25])
    elif plotting_corner_string == 'bottom_right':
        inset_axes_object = figure_object.add_axes([0.625, 0.3, 0.25, 0.25])
    elif plotting_corner_string == 'bottom_left':
        inset_axes_object = figure_object.add_axes([0.2, 0.3, 0.25, 0.25])
    elif plotting_corner_string == 'top_left':
        inset_axes_object = figure_object.add_axes([0.2, 0.55, 0.25, 0.25])

    nan_flags = numpy.logical_or(
        numpy.isnan(bin_mean_target_values),
        numpy.isnan(bin_mean_predictions)
    )
    assert not numpy.all(nan_flags)
    real_indices = numpy.where(numpy.invert(nan_flags))[0]

    target_handle = inset_axes_object.plot(
        bin_centers[real_indices], bin_mean_target_values[real_indices],
        color=MEAN_TARGET_LINE_COLOUR, linestyle='solid', linewidth=2
    )[0]

    prediction_handle = inset_axes_object.plot(
        bin_centers[real_indices], bin_mean_predictions[real_indices],
        color=MEAN_PREDICTION_LINE_COLOUR, linestyle='dashed', linewidth=2
    )[0]

    y_max = max([
        numpy.nanmax(bin_mean_predictions),
        numpy.nanmax(bin_mean_target_values)
    ])
    y_min = min([
        numpy.nanmin(bin_mean_predictions),
        numpy.nanmin(bin_mean_target_values)
    ])
    inset_axes_object.set_ylim(y_min, y_max)
    inset_axes_object.set_xlim(left=0.)

    for this_tick_object in inset_axes_object.xaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(INSET_FONT_SIZE)
        this_tick_object.label.set_rotation('vertical')

    for this_tick_object in inset_axes_object.yaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(INSET_FONT_SIZE)

    if for_spread_skill_plot:
        anchor_arg = (0.5, -0.25)
    else:
        anchor_arg = (0.5, -0.2)

    inset_axes_object.legend(
        [target_handle, prediction_handle],
        ['Mean target', 'Mean prediction'],
        loc='upper center', bbox_to_anchor=anchor_arg,
        fancybox=True, shadow=True, ncol=1, fontsize=INSET_FONT_SIZE
    )

    return inset_axes_object


def _plot_histogram(axes_object, bin_edges, bin_frequencies):
    """Plots histogram on existing axes.

    B = number of bins

    :param axes_object: Will plot histogram on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param bin_edges: length-(B + 1) numpy array with values at edges of each
        bin. These values will be plotted on the x-axis.
    :param bin_frequencies: length-B numpy array with fraction of examples in
        each bin. These values will be plotted on the y-axis.
    :return: histogram_axes_object: Axes handle for histogram only (also
        instance of `matplotlib.axes._subplots.AxesSubplot`).
    """

    histogram_axes_object = axes_object.twinx()
    axes_object.set_zorder(histogram_axes_object.get_zorder() + 1)
    axes_object.patch.set_visible(False)

    histogram_axes_object.bar(
        x=bin_edges[:-1], height=bin_frequencies, width=numpy.diff(bin_edges),
        color=INSET_HISTO_FACE_COLOUR, edgecolor=INSET_HISTO_EDGE_COLOUR,
        linewidth=INSET_HISTO_EDGE_WIDTH, align='edge'
    )

    return histogram_axes_object


def get_results_one_var(
        target_values, prediction_matrix, bin_edge_prediction_stdevs):
    """Computes spread-skill relationship for one target variable.

    E = number of examples
    S = number of ensemble members
    B = number of bins

    :param target_values: length-E numpy array of actual values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :param bin_edge_prediction_stdevs: length-(B - 1) numpy array of bin
        cutoffs.  Each is a standard deviation for the predictive distribution.
        Ultimately, there will be B + 1 edges; this method will use 0 as the
        lowest edge and inf as the highest edge.
    :return: result_dict: Dictionary with the following keys.
    result_dict['mean_prediction_stdevs']: length-B numpy array, where the [i]th
        entry is the mean standard deviation of predictive distributions in the
        [i]th bin.
    result_dict['bin_edge_prediction_stdevs']: length-(B + 1) numpy array,
        where the [i]th and [i + 1]th entries are the edges for the [i]th bin.
    result_dict['rmse_values']: length-B numpy array, where the [i]th
        entry is the root mean squared error of mean predictions in the [i]th
        bin.
    result_dict['spread_skill_reliability']: Spread-skill reliability (SSREL).
    result_dict['spread_skill_ratio']: Spread-skill ratio (SSRAT).
    result_dict['example_counts']: length-B numpy array of corresponding example
        counts.
    result_dict['mean_mean_predictions']: length-B numpy array, where the
        [i]th entry is the mean mean prediction for the [i]th bin.
    result_dict['mean_target_values']: length-B numpy array, where the [i]th
        entry is the mean target value for the [i]th bin.
    """

    # TODO: Generalize this method so that it doesn't rely on UQ taking the form
    # of an ensemble.

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

    assert len(bin_edge_prediction_stdevs.shape) == 1
    assert numpy.all(bin_edge_prediction_stdevs > 0.)
    assert numpy.all(numpy.diff(bin_edge_prediction_stdevs) > 0.)

    bin_edge_prediction_stdevs = numpy.concatenate((
        numpy.array([0.]),
        bin_edge_prediction_stdevs,
        numpy.array([numpy.inf])
    ))

    num_bins = len(bin_edge_prediction_stdevs) - 1
    assert num_bins >= 2

    # Do actual stuff.
    mean_predictions = numpy.mean(prediction_matrix, axis=1)
    predictive_stdevs = numpy.std(prediction_matrix, axis=1, ddof=1)
    squared_errors = (mean_predictions - target_values) ** 2

    mean_prediction_stdevs = numpy.full(num_bins, numpy.nan)
    rmse_values = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, 0, dtype=int)
    mean_mean_predictions = numpy.full(num_bins, numpy.nan)
    mean_target_values = numpy.full(num_bins, numpy.nan)

    for k in range(num_bins):
        these_indices = numpy.where(numpy.logical_and(
            predictive_stdevs >= bin_edge_prediction_stdevs[k],
            predictive_stdevs < bin_edge_prediction_stdevs[k + 1]
        ))[0]

        mean_prediction_stdevs[k] = numpy.sqrt(numpy.mean(
            predictive_stdevs[these_indices] ** 2
        ))
        rmse_values[k] = numpy.sqrt(numpy.mean(
            squared_errors[these_indices]
        ))

        example_counts[k] = len(these_indices)
        mean_mean_predictions[k] = numpy.mean(mean_predictions[these_indices])
        mean_target_values[k] = numpy.mean(target_values[these_indices])

    these_diffs = numpy.absolute(mean_prediction_stdevs - rmse_values)
    these_diffs[numpy.isnan(these_diffs)] = 0.
    spread_skill_reliability = numpy.average(
        these_diffs, weights=example_counts
    )

    this_numer = numpy.sqrt(numpy.mean(predictive_stdevs ** 2))
    this_denom = numpy.sqrt(numpy.mean(squared_errors))
    spread_skill_ratio = this_numer / this_denom

    return {
        MEAN_PREDICTION_STDEVS_KEY: mean_prediction_stdevs,
        BIN_EDGE_PREDICTION_STDEVS_KEY: bin_edge_prediction_stdevs,
        RMSE_VALUES_KEY: rmse_values,
        SPREAD_SKILL_RELIABILITY_KEY: spread_skill_reliability,
        SPREAD_SKILL_RATIO_KEY: spread_skill_ratio,
        EXAMPLE_COUNTS_KEY: example_counts,
        MEAN_MEAN_PREDICTIONS_KEY: mean_mean_predictions,
        MEAN_TARGET_VALUES_KEY: mean_target_values
    }


def plot_spread_vs_skill(
        result_dict, line_colour=DEFAULT_LINE_COLOUR,
        line_style='solid', line_width=DEFAULT_LINE_WIDTH):
    """Displays the spread-skill plot.

    :param result_dict: Dictionary returned by `get_results_one_var`.
    :param line_colour: Line colour (in any format accepted by matplotlib).
    :param line_style: Line style (in any format accepted by matplotlib).
    :param line_width: Line width (in any format accepted by matplotlib).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    mean_prediction_stdevs = result_dict[MEAN_PREDICTION_STDEVS_KEY]
    bin_edges = result_dict[BIN_EDGE_PREDICTION_STDEVS_KEY]
    rmse_values = result_dict[RMSE_VALUES_KEY]
    spread_skill_reliability = result_dict[SPREAD_SKILL_RELIABILITY_KEY]
    spread_skill_ratio = result_dict[SPREAD_SKILL_RATIO_KEY]
    example_counts = result_dict[EXAMPLE_COUNTS_KEY]
    mean_mean_predictions = result_dict[MEAN_MEAN_PREDICTIONS_KEY]
    mean_target_values = result_dict[MEAN_TARGET_VALUES_KEY]

    nan_flags = numpy.logical_or(
        numpy.isnan(mean_prediction_stdevs),
        numpy.isnan(rmse_values)
    )
    assert not numpy.all(nan_flags)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    max_value_to_plot = 1.01 * max([
        numpy.nanmax(mean_prediction_stdevs),
        numpy.nanmax(rmse_values)
    ])
    perfect_x_coords = numpy.array([0, max_value_to_plot])
    perfect_y_coords = numpy.array([0, max_value_to_plot])
    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=REFERENCE_LINE_COLOUR,
        linestyle='dashed', linewidth=REFERENCE_LINE_WIDTH
    )

    real_indices = numpy.where(numpy.invert(nan_flags))[0]
    axes_object.plot(
        mean_prediction_stdevs[real_indices],
        rmse_values[real_indices],
        color=line_colour, linestyle=line_style, linewidth=line_width,
        marker='o', markersize=12, markeredgewidth=0,
        markerfacecolor=line_colour, markeredgecolor=line_colour
    )

    unit_string = r's$^{-1}$'
    axes_object.set_xlabel(
        'Spread (stdev of predictive distribution; {0:s})'.format(unit_string)
    )
    axes_object.set_ylabel(
        'Skill (RMSE of mean prediction; {0:s})'.format(unit_string)
    )

    bin_frequencies = example_counts.astype(float) / numpy.sum(example_counts)

    if numpy.isnan(mean_prediction_stdevs[-1]):
        bin_edges[-1] = bin_edges[-2] + (bin_edges[-2] - bin_edges[-3])
    else:
        bin_edges[-1] = (
            bin_edges[-2] + 2 * (mean_prediction_stdevs[-1] - bin_edges[-2])
        )

    histogram_axes_object = _plot_histogram(
        axes_object=axes_object, bin_edges=bin_edges,
        bin_frequencies=bin_frequencies * 100
    )
    histogram_axes_object.set_ylabel('% examples in each bin')

    # axes_object.set_xlim(min([bin_edges[0], 0]), bin_edges[-1])
    # axes_object.set_ylim(0, 1.01 * numpy.nanmax(rmse_values))

    axes_object.set_xlim(min([bin_edges[0], 0]), max_value_to_plot)
    axes_object.set_ylim(0, max_value_to_plot)

    inset_axes_object = _plot_means_as_inset(
        figure_object=figure_object, bin_centers=mean_prediction_stdevs,
        bin_mean_predictions=mean_mean_predictions,
        bin_mean_target_values=mean_target_values,
        plotting_corner_string='bottom_right',
        for_spread_skill_plot=True
    )
    inset_axes_object.set_zorder(axes_object.get_zorder() + 1)

    inset_axes_object.set_xticks(axes_object.get_xticks())
    inset_axes_object.set_xlim(axes_object.get_xlim())
    inset_axes_object.set_xlabel(
        'Spread ({0:s})'.format(unit_string),
        fontsize=INSET_FONT_SIZE
    )
    inset_axes_object.set_ylabel(
        'Avg target or pred ({0:s})'.format(unit_string),
        fontsize=INSET_FONT_SIZE
    )
    inset_axes_object.set_title(
        'Avgs by model spread', fontsize=INSET_FONT_SIZE
    )

    title_string = (
        'Spread vs. skill for max future vorticity\n'
        'SSREL = {0:.6f} {1:s}; SSRAT = {2:.3f}'
    ).format(
        spread_skill_reliability,
        unit_string,
        spread_skill_ratio
    )

    axes_object.set_title(title_string)
    return figure_object, axes_object
