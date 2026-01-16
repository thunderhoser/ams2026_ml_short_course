"""Helper methods for any neural network (regardless of UQ approach)."""

import random
import numpy
import keras
import pandas
from ams2026_ml_short_course.utils import utils
from ams2026_ml_short_course.utils import image_utils
from ams2026_ml_short_course.utils import image_normalization


def data_generator(image_file_names, num_examples_per_batch,
                   normalization_dict):
    """Generates training or validation examples on the fly.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param image_file_names: 1-D list of paths to input files (readable by
        `image_utils.read_file`).
    :param num_examples_per_batch: Number of examples per training batch.
    :param normalization_dict: Dictionary with params used to normalize
        predictors.  See doc for `image_normalization.normalize_data`.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: target_values: length-E numpy array of target values (max future
        vorticity in s^-1).
    :raises: TypeError: if `normalization_dict is None`.
    """

    if normalization_dict is None:
        raise TypeError(
            'normalization_dict cannot be None.  Must be specified.'
        )

    random.shuffle(image_file_names)
    num_files = len(image_file_names)
    file_index = 0

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None
    predictor_names = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print('Reading data from: "{0:s}"...'.format(
                image_file_names[file_index]
            ))

            this_image_dict = image_utils.read_file(
                image_file_names[file_index]
            )
            predictor_names = this_image_dict[image_utils.PREDICTOR_NAMES_KEY]

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = (
                    this_image_dict[image_utils.PREDICTOR_MATRIX_KEY] + 0.
                )
                full_target_matrix = (
                    this_image_dict[image_utils.TARGET_MATRIX_KEY] + 0.
                )

            else:
                full_predictor_matrix = numpy.concatenate((
                    full_predictor_matrix,
                    this_image_dict[image_utils.PREDICTOR_MATRIX_KEY]
                ), axis=0)

                full_target_matrix = numpy.concatenate((
                    full_target_matrix,
                    this_image_dict[image_utils.TARGET_MATRIX_KEY]
                ), axis=0)

            num_examples_in_memory = full_target_matrix.shape[0]

        batch_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int
        )
        batch_indices = numpy.random.choice(
            batch_indices, size=num_examples_per_batch, replace=False
        )

        predictor_matrix, _ = image_normalization.normalize_data(
            predictor_matrix=full_predictor_matrix[batch_indices, ...],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict
        )
        target_values = numpy.max(
            full_target_matrix[batch_indices, ...],
            axis=(1, 2)
        )

        print((
            'Mean target value (max future vorticity in s^-1): {0:.4f}\n'
        ).format(
            numpy.mean(target_values)
        ))

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None

        yield predictor_matrix.astype('float32'), target_values


def create_data(image_file_names, normalization_dict):
    """Creates input data for CNN.

    This method is the same as `data_generator`, except that it returns all the
    data at once, rather than generating batches on the fly.

    :param image_file_names: See doc for `data_generator`.
    :param normalization_dict: Same.
    :return: predictor_matrix: Same.
    :return: target_values: Same.
    :raises: TypeError: if `normalization_dict is None`.
    """

    image_dict = image_utils.read_many_files(image_file_names)

    predictor_matrix, _ = image_normalization.normalize_data(
        predictor_matrix=image_dict[image_utils.PREDICTOR_MATRIX_KEY],
        predictor_names=image_dict[image_utils.PREDICTOR_NAMES_KEY],
        normalization_dict=normalization_dict
    )
    print('Mean predictor values = {0:s}'.format(
        str(numpy.mean(predictor_matrix, axis=(0, 1, 2)))
    ))

    target_values = numpy.max(
        image_dict[image_utils.TARGET_MATRIX_KEY],
        axis=(1, 2)
    )

    print((
        'Mean target value (max future vorticity in s^-1): {0:.4f}\n'
    ).format(
        numpy.mean(target_values)
    ))

    return predictor_matrix, target_values


def train_model_with_generator(
        model_object, training_file_names, validation_file_names,
        normalization_dict,
        num_epochs, num_examples_per_batch,
        num_training_batches_per_epoch, num_validation_batches_per_epoch,
        save_weights_every_epoch,
        output_dir_name):
    """Trains CNN with generator.

    :param model_object: Untrained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param training_file_names: 1-D list of paths to training files (readable by
        `image_utils.read_file`).
    :param validation_file_names: Same but for validation files.
    :param normalization_dict: See doc for `data_generator`.
    :param num_epochs: Number of epochs.
    :param num_examples_per_batch: Batch size.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param save_weights_every_epoch: Boolean flag.  If True, will save new model
        weights every epoch, regardless of validation loss.  If False, will save
        new weights only when validation loss reaches a new minimum.
    :param output_dir_name: Path to output directory (model will be saved here).
    """

    utils._mkdir_recursive_if_necessary(directory_name=output_dir_name)

    backup_dir_name = '{0:s}/backup_and_restore'.format(output_dir_name)
    utils._mkdir_recursive_if_necessary(directory_name=backup_dir_name)

    if save_weights_every_epoch:
        model_file_name = (
            '{0:s}/model_epoch={{epoch:04d}}_val-loss={{val_loss:.5f}}.'
            'weights.h5'
        ).format(output_dir_name)
    else:
        model_file_name = '{0:s}/model.weights.h5'.format(output_dir_name)

    history_file_name = '{0:s}/history.csv'.format(output_dir_name)

    try:
        history_table_pandas = pandas.read_csv(history_file_name)
        initial_epoch = history_table_pandas['epoch'].max() + 1
        best_validation_loss = history_table_pandas['val_loss'].min()
    except:
        initial_epoch = 0
        best_validation_loss = numpy.inf

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=True
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=not save_weights_every_epoch, save_weights_only=True,
        mode='min', save_freq='epoch'
    )
    checkpoint_object.best = best_validation_loss

    backup_object = keras.callbacks.BackupAndRestore(
        backup_dir_name, save_freq='epoch', delete_checkpoint=False
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.95,
        patience=10, verbose=1, mode='min',
        min_delta=0., cooldown=0
    )

    list_of_callback_objects = [
        history_object, checkpoint_object, backup_object, plateau_object
    ]

    training_generator = data_generator(
        image_file_names=training_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict
    )

    validation_generator = data_generator(
        image_file_names=validation_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict
    )

    model_object.fit(
        x=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def train_model_sans_generator(
        model_object, training_file_names, validation_file_names,
        normalization_dict,
        num_epochs, num_examples_per_batch,
        save_weights_every_epoch,
        output_dir_name):
    """Trains CNN without generator.

    :param model_object: See documentation for `train_model_with_generator`.
    :param training_file_names: Same.
    :param validation_file_names: Same.
    :param normalization_dict: Same.
    :param num_epochs: Same.
    :param num_examples_per_batch: Same.
    :param save_weights_every_epoch: Same.
    :param output_dir_name: Same.
    """

    utils._mkdir_recursive_if_necessary(directory_name=output_dir_name)

    backup_dir_name = '{0:s}/backup_and_restore'.format(output_dir_name)
    utils._mkdir_recursive_if_necessary(directory_name=backup_dir_name)

    if save_weights_every_epoch:
        model_file_name = (
            '{0:s}/model_epoch={{epoch:04d}}_val-loss={{val_loss:.5f}}.'
            'weights.h5'
        ).format(output_dir_name)
    else:
        model_file_name = '{0:s}/model.weights.h5'.format(output_dir_name)

    history_file_name = '{0:s}/history.csv'.format(output_dir_name)

    try:
        history_table_pandas = pandas.read_csv(history_file_name)
        initial_epoch = history_table_pandas['epoch'].max() + 1
        best_validation_loss = history_table_pandas['val_loss'].min()
    except:
        initial_epoch = 0
        best_validation_loss = numpy.inf

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=True
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=not save_weights_every_epoch, save_weights_only=True,
        mode='min', save_freq='epoch'
    )
    checkpoint_object.best = best_validation_loss

    backup_object = keras.callbacks.BackupAndRestore(
        backup_dir_name, save_freq='epoch', delete_checkpoint=False
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.95,
        patience=10, verbose=1, mode='min',
        min_delta=0., cooldown=0
    )

    list_of_callback_objects = [
        history_object, checkpoint_object, backup_object, plateau_object
    ]

    training_predictor_matrix, training_target_values = create_data(
        image_file_names=training_file_names,
        normalization_dict=normalization_dict
    )
    training_max_refls = numpy.max(training_predictor_matrix[..., 0], axis=(1, 2))
    print(numpy.corrcoef(training_max_refls, training_target_values))

    validation_predictor_matrix, validation_target_values = create_data(
        image_file_names=validation_file_names,
        normalization_dict=normalization_dict
    )
    validation_max_refls = numpy.max(validation_predictor_matrix[..., 0], axis=(1, 2))
    print(numpy.corrcoef(validation_max_refls, validation_target_values))

    model_object.fit(
        x=training_predictor_matrix,
        y=training_target_values,
        batch_size=num_examples_per_batch,
        steps_per_epoch=None,
        epochs=num_epochs,
        shuffle=True,
        initial_epoch=initial_epoch,
        verbose=1,
        callbacks=list_of_callback_objects,
        validation_data=(validation_predictor_matrix, validation_target_values),
        validation_steps=None
    )


def apply_model(model_object, predictor_matrix, verbose=True):
    """Applies trained CNN to new data.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    S = number of predictions per data examples

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: prediction_matrix: E-by-S numpy array of predictions.
    """

    num_examples = predictor_matrix.shape[0]
    num_examples_per_batch = 1000
    prediction_matrix = None

    for i in range(0, num_examples, num_examples_per_batch):
        first_index = i
        last_index = min([i + num_examples_per_batch, num_examples])

        if verbose:
            print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
                first_index + 1, last_index, num_examples
            ))

        this_prediction_matrix = model_object.predict(
            predictor_matrix[first_index:last_index, ...],
            batch_size=last_index - first_index
        )

        if prediction_matrix is None:
            these_dim = (num_examples,) + this_prediction_matrix.shape[1:]
            prediction_matrix = numpy.full(these_dim, numpy.nan)

        prediction_matrix[first_index:last_index, ...] = this_prediction_matrix

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    return prediction_matrix


def read_model_weights(model_object, hdf5_file_name):
    """Reads model weights from HDF5 file.

    :param model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`, needed to specify the architecture.
    :param hdf5_file_name: Path to HDF5 file with trained weights.
    :return: model_object: Same as input but with different weights.
    """

    model_object.load_weights(hdf5_file_name)
    return model_object


def convert_pdp_to_ensemble(prediction_matrix, ensemble_size):
    """Converts parametric distributional prediction (PDP) to ensemble.

    E = number of examples (storm objects)
    S = ensemble size

    :param prediction_matrix: E-by-2 numpy array of predictions, where
        prediction_matrix[:, 0] contains means and prediction_matrix[:, 1]
        contains standard deviations.
    :param ensemble_size: Number of ensemble members.
    :return: prediction_matrix: E-by-S numpy array of predictions.
    """

    assert len(prediction_matrix.shape) == 2
    assert prediction_matrix.shape[1] == 2
    assert numpy.all(prediction_matrix >= 0.)

    num_examples = prediction_matrix.shape[0]
    predicted_means = prediction_matrix[:, 0]
    predicted_stdevs = prediction_matrix[:, 1]

    prediction_matrix = numpy.random.normal(
        loc=predicted_means[:, numpy.newaxis],
        scale=predicted_stdevs[:, numpy.newaxis],
        size=(num_examples, ensemble_size)
    )
    prediction_matrix = numpy.maximum(prediction_matrix, 0.)

    return prediction_matrix
