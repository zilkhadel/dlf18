import os
import numpy as np
import datetime as dt

from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, classification_report
from cnn_finetune.vgg16 import vgg16_model

from utils.shortcuts import pj, pe, ps, dump, Paths
from utils.reader import gen_exp_data_dir, get_train_and_valid_generators
from utils.saver import WeightsSaver


def run_experiemnt(gender,
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
    Train a model on a pair of subjects, according to speficied arguments and get predictions on validation set, as well as performance metrics.
    :param gender: the gender of the subjects for which to create an experiment
    :param train_samples: number of training samples
    :param validation_samples: number of validation samples
    :param subjects: [optional] a tuple of exactly two subject names, of the same gender. if not set, two random subjects of the same gender will be used.
    :param img_size: [optional] height / width of input images
    :param img_channels: [optional] number of color channels in input images
    :param num_classes: [optional] number of ouput classes
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
    exp_name, exp_data_dir, actual_train_samples, actual_valid_samples = gen_exp_data_dir(gender, train_samples, validation_samples, subjects)

    # get training and validation data generators
    train_generator, validation_generator = get_train_and_valid_generators(exp_data_dir, batch_size, img_size)

    # load vgg16 model
    # initial_weights_path = pj(Paths.pretrained_dir, 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    initial_weights_path = pj(Paths.experiments_dir, 'Laurent_Alexis_2018-02-09_2222', 'Laurent_Alexis_2018-02-09_2222_weights_final.h5')
    initial_weights_num_classes = 2
    initial_weights = (initial_weights_path, initial_weights_num_classes)

    metrics = ['accuracy']  # accuracy_score, precision_score, recall_score, f1_score
    model = vgg16_model(img_size, img_size, img_channels, num_classes, initial_weights, freeze_first_layers, learning_rate, metrics)

    # # start fine-tuning the model
    model.fit_generator(train_generator,
                        epochs=epochs,
                        verbose=1,
                        validation_data=validation_generator,
                        callbacks=[WeightsSaver(model, save_each, exp_data_dir)])

    # save final model weights to disk
    final_weights_path = pj(exp_data_dir, f'{exp_name}_weights_final.h5')
    model.save_weights(final_weights_path)

    # make predictions on the validation/test set
    validation_predictions = model.predict_generator(validation_generator, verbose=1)
    validation_y_pred = np.rint(validation_predictions)
    validation_y_true = np.array([[1-yt, yt] for yt in validation_generator.classes])  # Important: validation generator must be used with shuffle=False for this to work.

    # generate predictions object and save it to exp_data_dir
    predictions_data_path = pj(exp_data_dir, f'{exp_name}_validations_predictions.csv')
    predictions_data = [('class', 'filename', 'y_true', 'y_pred', 'prob_0', 'prob_1')]
    predictions_data += [(validation_generator.class_indices[ps(validation_generator.filenames[ind])[0]],
                          ps(validation_generator.filenames[ind])[1],
                          validation_y_true[ind][1],
                          int(validation_y_pred[ind][1]),
                          pred[0],
                          pred[1]) for ind, pred in enumerate(validation_predictions)]
    dump(predictions_data, predictions_data_path)

    # cross-entropy loss score on the validation/test set
    validation_loss = log_loss(validation_y_true, validation_y_pred)
    validation_accuracy = accuracy_score(validation_y_true, validation_y_pred)
    validation_precision = precision_score(validation_y_true, validation_y_pred, average=None)
    validation_recall = recall_score(validation_y_true, validation_y_pred, average=None)
    validation_f1 = f1_score(validation_y_true, validation_y_pred, average=None)

    # save classification report
    report = classification_report(validation_y_true, validation_y_pred, list(validation_generator.class_indices.values()), list(validation_generator.class_indices.keys()))
    print(report)
    dump(report, pj(exp_data_dir, 'report.txt'))

    # log end and run times
    end_time = dt.datetime.now()
    print(f'End time: {end_time}')
    print(f'Run time: {end_time - start_time}')

    # save experiment statistics to disk
    exp_stats = {'Exp name:': exp_name,
                 'Start time:': start_time,
                 'End time:': end_time,
                 'Run time:': (end_time - start_time),
                 'Validation Loss:': validation_loss,
                 'Validation Accuracy:': validation_accuracy,
                 'Validation Precision:': validation_precision,
                 'Validation Recall:': validation_recall,
                 'Validation F1:': validation_f1,
                 '': ''}

    exp_stats_path = pj(exp_data_dir, f'{exp_name}.log')
    dump(exp_stats.items(), exp_stats_path, append=True)

    # if final weights are saved, delete intermediate weights file
    intermediate_weights_path = pj(exp_data_dir, 'weights.h5')
    if pe(final_weights_path) and pe(intermediate_weights_path):
        os.remove(intermediate_weights_path)
