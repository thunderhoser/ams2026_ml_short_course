"""Helper methods for computing PIT values and PIT histogram.

PIT = probability-integral transform
"""

import numpy
from matplotlib import pyplot
from scipy.stats import percentileofscore

TOLERANCE = 1e-6

MAX_PIT_FOR_LOW_BINS = 0.3
MIN_PIT_FOR_HIGH_BINS = 0.7
CONFIDENCE_LEVEL_FOR_NONEXTREME_PIT = 0.95

BIN_EDGES_KEY = 'bin_edges'
BIN_COUNTS_KEY = 'bin_counts'
PITD_KEY = 'pitd_value'
PERFECT_PITD_KEY = 'perfect_pitd_value'
LOW_BIN_BIAS_KEY = 'low_bin_pit_bias'
MIDDLE_BIN_BIAS_KEY = 'middle_bin_pit_bias'
HIGH_BIN_BIAS_KEY = 'high_bin_pit_bias'
EXTREME_PIT_FREQ_KEY = 'extreme_pit_frequency'

DEFAULT_HISTOGRAM_FACE_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
DEFAULT_HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
DEFAULT_HISTOGRAM_EDGE_WIDTH = 2.

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
REFERENCE_LINE_WIDTH = 2.

FIGURE_WIDTH_INCHES = 10
FIGURE_HEIGHT_INCHES = 10

DEFAULT_FONT_SIZE = 24
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)


def _get_low_mid_hi_bins(bin_edges):
    """Returns indices for low-PIT, medium-PIT, and high-PIT bins.

    B = number of bins

    :param bin_edges: length-(B + 1) numpy array of bin edges, sorted in
        ascending order.
    :return: low_bin_indices: 1-D numpy array with array indices for low-PIT
        bins.
    :return: middle_bin_indices: 1-D numpy array with array indices for
        medium-PIT bins.
    :return: high_bin_indices: 1-D numpy array with array indices for high-PIT
        bins.
    """

    num_bins = len(bin_edges) - 1

    these_diffs = bin_edges - MAX_PIT_FOR_LOW_BINS
    these_diffs[these_diffs > TOLERANCE] = numpy.inf
    max_index_for_low_bins = numpy.argmin(numpy.absolute(these_diffs)) - 1
    max_index_for_low_bins = max([max_index_for_low_bins, 0])

    low_bin_indices = numpy.linspace(
        0, max_index_for_low_bins, num=max_index_for_low_bins + 1, dtype=int
    )

    these_diffs = MIN_PIT_FOR_HIGH_BINS - bin_edges
    these_diffs[these_diffs > TOLERANCE] = numpy.inf
    min_index_for_high_bins = numpy.argmin(numpy.absolute(these_diffs))
    min_index_for_high_bins = min([min_index_for_high_bins, num_bins - 1])

    high_bin_indices = numpy.linspace(
        min_index_for_high_bins, num_bins - 1,
        num=num_bins - min_index_for_high_bins, dtype=int
    )

    middle_bin_indices = numpy.linspace(
        0, num_bins - 1, num=num_bins, dtype=int
    )
    middle_bin_indices = numpy.array(list(
        set(middle_bin_indices.tolist())
        - set(low_bin_indices.tolist())
        - set(high_bin_indices.tolist())
    ))

    return low_bin_indices, middle_bin_indices, high_bin_indices


def get_histogram_one_var(target_values, prediction_matrix, num_bins):
    """Computes PIT histogram for one variable.

    E = number of examples
    S = number of ensemble members
    B = number of bins in histogram

    :param target_values: length-E numpy array of actual values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :param num_bins: Number of bins in histogram.
    :return: result_dict: Dictionary with the following keys.
    result_dict["bin_edges"]: length-(B + 1) numpy array of bin edges (ranging
        from 0...1, because PIT ranges from 0...1).
    result_dict["bin_counts"]: length-B numpy array with number of examples in
        each bin.
    result_dict["pitd_value"]: Value of the calibration-deviation metric (PITD).
    result_dict["perfect_pitd_value"]: Minimum expected PITD value.
    result_dict["low_bin_pit_bias"]: PIT bias for low bins, i.e., PIT values of
        [0, 0.3).
    result_dict["middle_bin_pit_bias"]: PIT bias for middle bins, i.e., PIT
        values of [0.3, 0.7).
    result_dict["high_bin_pit_bias"]: PIT bias for high bins, i.e., PIT values
        of [0.7, 1.0].
    result_dict["extreme_pit_frequency"]: Frequency of extreme PIT values, i.e.,
        below 0.025 or above 0.975.
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

    num_bins = int(numpy.round(num_bins))
    assert num_bins >= 2

    # Do actual stuff.
    num_examples = len(target_values)
    pit_values = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        pit_values[i] = 0.01 * percentileofscore(
            a=prediction_matrix[i, :], score=target_values[i], kind='mean'
        )

    bin_edges = numpy.linspace(0, 1, num=num_bins + 1, dtype=float)
    indices_example_to_bin = numpy.digitize(
        x=pit_values, bins=bin_edges, right=False
    ) - 1
    indices_example_to_bin[indices_example_to_bin < 0] = 0
    indices_example_to_bin[indices_example_to_bin >= num_bins] = num_bins - 1

    used_bin_indices, used_bin_counts = numpy.unique(
        indices_example_to_bin, return_counts=True
    )
    bin_counts = numpy.full(num_bins, 0, dtype=int)
    bin_counts[used_bin_indices] = used_bin_counts

    bin_frequencies = bin_counts.astype(float) / num_examples
    perfect_bin_frequency = 1. / num_bins

    pitd_value = numpy.sqrt(
        numpy.mean((bin_frequencies - perfect_bin_frequency) ** 2)
    )
    perfect_pitd_value = numpy.sqrt(
        (1. - perfect_bin_frequency) / (num_examples * num_bins)
    )

    low_bin_indices, middle_bin_indices, high_bin_indices = (
        _get_low_mid_hi_bins(bin_edges)
    )

    low_bin_pit_bias = numpy.mean(
        bin_frequencies[low_bin_indices] - perfect_bin_frequency
    )
    middle_bin_pit_bias = numpy.mean(
        bin_frequencies[middle_bin_indices] - perfect_bin_frequency
    )
    high_bin_pit_bias = numpy.mean(
        bin_frequencies[high_bin_indices] - perfect_bin_frequency
    )
    extreme_pit_frequency = numpy.mean(numpy.logical_or(
        pit_values < 0.5 * (1. - CONFIDENCE_LEVEL_FOR_NONEXTREME_PIT),
        pit_values > 0.5 * (1. + CONFIDENCE_LEVEL_FOR_NONEXTREME_PIT)
    ))

    return {
        BIN_EDGES_KEY: bin_edges,
        BIN_COUNTS_KEY: bin_counts,
        PITD_KEY: pitd_value,
        PERFECT_PITD_KEY: perfect_pitd_value,
        LOW_BIN_BIAS_KEY: low_bin_pit_bias,
        MIDDLE_BIN_BIAS_KEY: middle_bin_pit_bias,
        HIGH_BIN_BIAS_KEY: high_bin_pit_bias,
        EXTREME_PIT_FREQ_KEY: extreme_pit_frequency
    }


def plot_pit_histogram(
        result_dict, model_description_string,
        face_colour=DEFAULT_HISTOGRAM_FACE_COLOUR,
        edge_colour=DEFAULT_HISTOGRAM_EDGE_COLOUR,
        edge_width=DEFAULT_HISTOGRAM_EDGE_WIDTH):
    """Plots PIT histogram for one target variable.

    :param result_dict: Dictionary returned by `get_histogram_one_var`.
    :param model_description_string: Model or dataset description (will go at
        beginning of title).
    :param face_colour: Face colour (in any format accepted by matplotlib).
    :param edge_colour: Edge colour (in any format accepted by matplotlib).
    :param edge_width: Edge width (in any format accepted by matplotlib).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    bin_edges = result_dict[BIN_EDGES_KEY]
    bin_counts = result_dict[BIN_COUNTS_KEY]
    pitd_value = result_dict[PITD_KEY]
    perfect_pitd_value = result_dict[PERFECT_PITD_KEY]
    low_bin_pit_bias = result_dict[LOW_BIN_BIAS_KEY]
    middle_bin_pit_bias = result_dict[MIDDLE_BIN_BIAS_KEY]
    high_bin_pit_bias = result_dict[HIGH_BIN_BIAS_KEY]
    extreme_pit_frequency = result_dict[EXTREME_PIT_FREQ_KEY]

    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.bar(
        x=bin_edges[:-1], height=bin_frequencies, width=numpy.diff(bin_edges),
        color=face_colour, edgecolor=edge_colour, linewidth=edge_width,
        align='edge'
    )

    num_bins = len(bin_edges) - 1
    perfect_x_coords = numpy.array([0, 1], dtype=float)
    perfect_y_coords = numpy.array([1. / num_bins, 1. / num_bins])
    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=REFERENCE_LINE_COLOUR,
        linestyle='dashed', linewidth=REFERENCE_LINE_WIDTH
    )

    axes_object.set_xlabel('PIT value')
    axes_object.set_ylabel('Frequency')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(bottom=0.)

    title_string = (
        '{0:s}\n'
        'PIT histogram for max future vorticity\n'
        'PITD = {1:.4f}; perfect PITD = {2:.4f}; extreme-PIT freq = {3:.4f}\n'
        '(Low-, middle-, high-)bin bias = ({4:.4f}, {5:.4f}, {6:.4f})'
    ).format(
        model_description_string,
        pitd_value,
        perfect_pitd_value,
        extreme_pit_frequency,
        low_bin_pit_bias,
        middle_bin_pit_bias,
        high_bin_pit_bias
    )

    axes_object.set_title(title_string)

    return figure_object, axes_object
