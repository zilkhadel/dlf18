"""
    Experiment Name:        vgg16_cifar_frz36

    Experiment Description:
        training vgg16 model on cifar with first 36 layer frozen (training only softmax layer).

    Experiment Details:
        model:              vgg16
        total layers:       37
        frozen layers:      36
        learning rate:      0.001
        train set size:     3000
        test set size:      100
        batch size:         16
        epochs:             10

    Experiment Results:
        accuracy:           ???

"""

import os
import datetime as dt
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score  # , classification_report

from cnn_finetune.vgg16 import vgg16_model
from cnn_finetune.load_cifar10 import load_cifar10_data

from utils.shortcuts import pj, pe, mkdirs, dump, objdump, Paths
from utils.saver import WeightsSaver
# from utils.metrics import get_precision, get_recall, get_f1

EXP_NAME = 'vgg16_cifar_frz36'

if __name__ == '__main__':

    start_time = dt.datetime.now()
    print(f'Start time: {start_time}')

    img_rows, img_cols = 224, 224   # height and width of input images
    img_channels = 3                # number of color channels in input images
    num_classes = 10                # number of ouput classes
    train_samples = 50  # 3000      # number of training samples
    valid_samples = 10  # 100       # number of validation samples
    batch_size = 25     # 16        # number of training samples per gradient update
    epochs = 1          # 10        # number of iteration over the entire training set
    freeze_first_layers = 36        # number of first layers to freeze
    save_each = 5                   # number of batches after which to save weights
    learning_rate = 0.001           # the step to use in each gradient update

    exp_dir = pj(Paths.experiments_dir, EXP_NAME)
    mkdirs(exp_dir)

    # load cifar10 training and validation/test data
    x_train, y_train, x_valid, y_valid = load_cifar10_data(img_rows, img_cols, train_samples, valid_samples)

    # load vgg16 model
    initial_weights_path = pj(Paths.pretrained_dir, 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    initial_weights_num_classes = 1000
    initial_weights = (initial_weights_path, initial_weights_num_classes)
    metrics = ['accuracy']  # , get_precision, get_recall, get_f1
    model = vgg16_model(img_rows, img_cols, img_channels, num_classes, initial_weights, freeze_first_layers, learning_rate, metrics)

    # start fine-tuning the model
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[WeightsSaver(model, save_each, exp_dir)])

    # save final model weights to disk
    final_weights_path = pj(exp_dir, f'{EXP_NAME}_weights_final.h5')
    model.save_weights(final_weights_path)

    # make predictions on the validation/test set
    p_valid = model.predict(x_valid, batch_size=batch_size, verbose=1)

    # cross-entropy loss score on the validation/test set
    loss_valid = log_loss(y_valid, p_valid)

    # generate predictions object and save it to exp_data_dir
    predictions_data_path = pj(exp_dir, f'{EXP_NAME}_validations_predictions.csv')
    predictions_data = [('y_true', 'y_pred', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4', 'prob_5', 'prob_6', 'prob_7', 'prob_8', 'prob_9')]
    predictions_data += [(np.argmax(y_valid[ind]), np.argmax(p_valid[ind]), pred[0], pred[1], pred[2], pred[3], pred[4], pred[5], pred[6], pred[7], pred[8], pred[9]) for ind, pred in enumerate(p_valid)]
    dump(predictions_data, predictions_data_path, delimiter=',')

    # cross-entropy loss score on the validation/test set
    # p_valid_one_hot = np.array([[int(i == np.argmax(pv)) for i in range(0, len(pv))] for pv in p_valid], dtype=np.float32)  # convert the probabilities matrix to an array of 1-hot vectors.
    yv = [np.argmax(y) for y in y_valid]
    pv = [np.argmax(p) for p in p_valid]
    objdump([y_valid, p_valid, yv, pv], pj(exp_dir, f'{EXP_NAME}_validations_predictions.pkl'))

    validation_loss = log_loss(y_valid, p_valid)
    validation_accuracy = accuracy_score(yv, pv)
    validation_precision = precision_score(yv, pv, average='micro')
    validation_recall = recall_score(yv, pv, average='micro')
    validation_f1 = f1_score(yv, pv, average='micro')

    # save classification report
    # report = classification_report(y_valid, p_valid)
    # print(report)
    # dump(report, pj(exp_dir, 'report.txt'))

    # log end and run times
    end_time = dt.datetime.now()
    print(f'End time: {end_time}')
    print(f'Run time: {end_time - start_time}')

    # save experiment statistics to disk
    exp_stats = {'Exp name:': EXP_NAME,
                 'Start time:': start_time,
                 'End time:': end_time,
                 'Run time:': (end_time - start_time),
                 'Validation Loss:': validation_loss,
                 'Validation Accuracy:': validation_accuracy,
                 'Validation Precision:': validation_precision,
                 'Validation Recall:': validation_recall,
                 'Validation F1:': validation_f1,
                 '': ''}  # TODO : add more statistics to the log file.

    exp_stats_path = pj(exp_dir, f'{EXP_NAME}.log')
    dump(exp_stats.items(), exp_stats_path, append=True)

    # if final weights are saved, delete intermediate weights file
    intermediate_weights_path = pj(exp_dir, 'weights.h5')
    if pe(final_weights_path) and pe(intermediate_weights_path):
        os.remove(intermediate_weights_path)
