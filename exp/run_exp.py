import os
import math
import numpy as np
import datetime as dt

from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, classification_report
from cnn_finetune.vgg16 import vgg16_model

from utils.shortcuts import pj, pe, ps, dump, objdump, Paths
from utils.reader import gen_exp_data_dir, get_train_and_valid_generators
from utils.saver import WeightsSaver


def run_experiment(gender,
                   train_samples,
                   validation_samples,
                   subjects=None,
                   img_size=224,
                   img_channels=3,
                   num_classes=2,
                   batch_size=16,
                   epochs=10,
                   freeze_first_layers=36,
                   save_each=5,
                   learning_rate=0.001):
    """
    Train a model on a pair of subjects, according to specified arguments and get predictions on validation set, as well as performance metrics.
    :param gender: the gender of the subjects for which to create an experiment
    :param train_samples: number of training samples
    :param validation_samples: number of validation samples
    :param subjects: [optional] a tuple of exactly two subject names, of the same gender. if not set, two random subjects of the same gender will be used.
    :param img_size: [optional] height / width of input images
    :param img_channels: [optional] number of color channels in input images
    :param num_classes: [optional] number of output classes
    :param batch_size: [optional] number of training samples per gradient update
    :param epochs: [optional] number of iteration over the entire training set
    :param freeze_first_layers: [optional] number of layers to freeze
    :param save_each: [optional] number of batches after which to save intermediate weights h5 file
    :param learning_rate: [optional] the step to use in each gradient update
    :return:
    """

    # print start time
    start_time = dt.datetime.now()
    print(f'Start time: {start_time}')

    # generate data for experiment
    print(f'Generating data for experiment | Gender: {gender} | Requested train samples: {train_samples} | Requested validation samples: {validation_samples} | Subjects: {subjects}')
    exp_name, exp_data_dir, actual_train_samples, actual_validation_samples = gen_exp_data_dir(gender, train_samples, validation_samples, subjects)
    print(f'Generated data for experiment | Exp name: {exp_name} | Actual train samples: {actual_train_samples} | Actual validation samples: {actual_validation_samples} | Exp dir: {exp_data_dir}')

    # get training and validation data generators
    print(f'Getting train and validation generators | Batch size: {batch_size} | Image size: {img_size}')
    train_generator, validation_generator = get_train_and_valid_generators(exp_data_dir, batch_size, img_size)

    # load vgg16 model
    initial_weights_path = pj(Paths.pretrained_dir, 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    initial_weights_num_classes = 1000
    initial_weights = (initial_weights_path, initial_weights_num_classes)

    print(f'Loading vgg16 model | Initial weights path: {initial_weights_path} | Initial weights number of classes: {initial_weights_num_classes}')
    metrics = ['accuracy']
    model = vgg16_model(img_size, img_size, img_channels, num_classes, initial_weights, freeze_first_layers, learning_rate, metrics)

    # start fine-tuning the model
    print(f'Training model | Epochs: {epochs}')
    model.fit_generator(train_generator,
                        epochs=epochs,
                        verbose=1,
                        validation_data=validation_generator,
                        callbacks=[WeightsSaver(model, save_each, exp_data_dir)])

    # save final model weights to disk
    final_weights_path = pj(exp_data_dir, f'{exp_name}_weights_final.h5')
    print(f'Saving model weights | Path: {final_weights_path}')
    model.save_weights(final_weights_path)

    # make predictions on the validation/test set
    print('Making predictions on validation set')
    # make predictions on validation set
    validation_predictions = model.predict_generator(validation_generator, verbose=1)

    # convert the probabilities matrix to an array of predicted classes (since it's binary classification, it's the same as 1-hot vectors).
    validation_y_pred = np.array([np.argmax(p) for p in validation_predictions], dtype=np.float32)

    # generate an array of true classes. Important: validation generator must be used with shuffle=False for this to work.
    validation_y_true = validation_generator.classes  # np.array([[1-yt, yt] for yt in validation_generator.classes], dtype=np.float32)

    # generate predictions object and save it to exp_data_dir
    predictions_data_path = pj(exp_data_dir, f'{exp_name}_validations_predictions.csv')
    print(f'Saving predictions on validation set | Path: {predictions_data_path}')
    predictions_data = [('class', 'filename', 'y_true', 'y_pred', 'prob_0', 'prob_1')]
    predictions_data += [(validation_generator.class_indices[ps(validation_generator.filenames[ind])[0]],
                          ps(validation_generator.filenames[ind])[1],
                          validation_y_true[ind],
                          int(validation_y_pred[ind]),
                          pred[0],
                          pred[1]) for ind, pred in enumerate(validation_predictions)]
    dump(predictions_data, predictions_data_path, delimiter=',')
    objdump([validation_y_true, validation_y_pred], pj(exp_data_dir, f'{exp_name}_validations_predictions.pkl'))

    # cross-entropy loss score on the validation/test set
    print('Getting metrics on validation set predictions')
    validation_loss = log_loss(validation_y_true, validation_y_pred)
    validation_accuracy = accuracy_score(validation_y_true, validation_y_pred)
    validation_precision = precision_score(validation_y_true, validation_y_pred, average='micro')
    validation_recall = recall_score(validation_y_true, validation_y_pred, average='micro')
    validation_f1 = f1_score(validation_y_true, validation_y_pred, average='micro')

    # save classification report
    report = classification_report(validation_y_true, validation_y_pred, list(validation_generator.class_indices.values()), list(validation_generator.class_indices.keys()))
    print(report)
    dump(report, pj(exp_data_dir, 'report.txt'))

    # log end and run times
    end_time = dt.datetime.now()
    print(f'End time: {end_time}')
    print(f'Run time: {end_time - start_time}')

    # save experiment statistics to disk
    exp_stats = {'Gender:': gender,
                 'Exp name:': exp_name,
                 'Start time:': start_time,
                 'End time:': end_time,
                 'Run time:': (end_time - start_time),
                 '-': '-',
                 'Train samples:': train_samples,
                 'Epochs:': epochs,
                 'Batch size:': batch_size,
                 'Steps per epoch:': math.ceil((train_samples * num_classes) / batch_size),
                 'Freeze first layers:': freeze_first_layers,
                 'Learning rate:': learning_rate,
                 'Save each:': save_each,
                 '--': '--',
                 'Validation samples:': validation_samples,
                 'Validation loss:': validation_loss,
                 'Validation accuracy:': validation_accuracy,
                 'Validation precision:': validation_precision,
                 'Validation recall:': validation_recall,
                 'Validation F1:': validation_f1,
                 '': ''}

    exp_stats_path = pj(exp_data_dir, f'{exp_name}.log')
    dump(exp_stats.items(), exp_stats_path, append=True)

    # if final weights are saved, delete intermediate weights file
    intermediate_weights_path = pj(exp_data_dir, 'weights.h5')
    if pe(final_weights_path) and pe(intermediate_weights_path):
        os.remove(intermediate_weights_path)
