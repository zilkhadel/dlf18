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

import datetime as dt
from sklearn.metrics import log_loss

from cnn_finetune.vgg16 import vgg16_model
from cnn_finetune.load_cifar10 import load_cifar10_data

from utils.shortcuts import pj, mkdirs, dump, Paths
from utils.saver import WeightsSaver


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
    epochs = 2          # 10        # number of iteration over the entire training set
    save_each = 5                   # number of batches after which to save weights

    exp_dir = pj(Paths.experiments_dir, EXP_NAME)
    mkdirs(exp_dir)

    # load cifar10 training and validation/test data
    x_train, y_train, x_valid, y_valid = load_cifar10_data(img_rows, img_cols, train_samples, valid_samples)

    # load vgg16 model
    model = vgg16_model(img_rows, img_cols, img_channels, num_classes)

    # start tine-tuning the model
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[WeightsSaver(model, save_each, exp_dir)])

    # save final model weights to disk
    model.save_weights(pj(exp_dir, f'{EXP_NAME}_weights_final.h5'))

    # make predictions on the validation/test set
    p_valid = model.predict(x_valid, batch_size=batch_size, verbose=1)

    # cross-entropy loss score on the validation/test set
    loss_valid = log_loss(y_valid, p_valid)

    # save model predictions to disk
    p_valid_path = pj(exp_dir, f'{EXP_NAME}_pred_valid.csv')
    dump(p_valid, p_valid_path)

    # log end and run times
    end_time = dt.datetime.now()
    print(f'End time: {end_time}')
    print(f'Run time: {end_time - start_time}')

    # save experiment statistics to disk
    exp_stats = {'Exp name:': EXP_NAME,
                 'Start time:': start_time,
                 'End time:': end_time,
                 'Run time:': (end_time - start_time),
                 'Loss (valid):': loss_valid}  # TODO : add more statistics to the log file.

    exp_stats_path = pj(exp_dir, f'{EXP_NAME}.log')
    dump(exp_stats.items(), exp_stats_path)
