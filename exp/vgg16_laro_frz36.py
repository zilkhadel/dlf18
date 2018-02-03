"""
    Experiment Name:        vgg16_laro_frz36

    Experiment Description:
        training vgg16 model on laurent-romain dataset with first 36 layer frozen (training only softmax layer).

    Experiment Details:
        model:              vgg16
        total layers:       37
        frozen layers:      36
        learning rate:      0.001
        train set size:     260
        valid set size:     70
        batch size:         32
        epochs:             10

    Experiment Results:
        accuracy:           ???

"""

import datetime as dt
from sklearn.metrics import log_loss

from cnn_finetune.vgg16 import vgg16_model

from utils.shortcuts import pj, mkdirs, dump, Paths
from utils.reader import get_train_and_valid_generators
from utils.saver import WeightsSaver
from utils.metrics import get_f1


EXP_NAME = 'vgg16_laro_frz36'

if __name__ == '__main__':

    start_time = dt.datetime.now()
    print(f'Start time: {start_time}')

    img_rows, img_cols = 224, 224   # height and width of input images
    img_channels = 3                # number of color channels in input images
    num_classes = 2                 # number of ouput classes
    train_samples = 260             # number of training samples
    valid_samples = 60              # number of validation samples
    batch_size = 20                 # number of training samples per gradient update
    epochs = 2                      # number of iteration over the entire training set
    steps_per_epoch = 13            # number of steps (batches of samples) to perform before declaring one epoch finished and starting the next epoch (typically equals train_samples / batch_size).
    valid_steps_per_epoch = 3       # number of steps (batches of samples) to perform before declaring one epoch finished and starting the next epoch (typically equals valid_samples / batch_size).
    freeze_first_layers = 36        # number of first layers to freeze
    save_each = 5                   # number of batches after which to save weights
    learning_rate = 0.001           # the step to use in each gradient update

    exp_dir = pj(Paths.experiments_dir, EXP_NAME)
    mkdirs(exp_dir)

    # get training and validation data generators
    train_data_dir = pj(Paths.data_dir, 'laurent-romain', 'train')
    validation_data_dir = pj(Paths.data_dir, 'laurent-romain', 'validation')
    train_generator, validation_generator = get_train_and_valid_generators(train_data_dir, validation_data_dir, batch_size)

    # load vgg16 model
    initial_weights_path = pj(Paths.pretrained_dir, 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    metrics = ['accuracy', get_f1]
    model = vgg16_model(img_rows, img_cols, img_channels, num_classes, initial_weights_path, freeze_first_layers, learning_rate, metrics)

    # start fine-tuning the model
    model.fit_generator(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        validation_data=validation_generator,
                        validation_steps=valid_steps_per_epoch,
                        callbacks=[WeightsSaver(model, save_each, exp_dir)])

    # save final model weights to disk
    model.save_weights(pj(exp_dir, f'{EXP_NAME}_weights_final.h5'))

    # make predictions on the validation/test set
    p_validation = model.predict_generator(validation_generator, verbose=1)

    # cross-entropy loss score on the validation/test set
    # loss_valid = log_loss(validation_generator, p_validation)

    # save model predictions to disk
    p_valid_path = pj(exp_dir, f'{EXP_NAME}_pred_valid.csv')
    dump(p_validation, p_valid_path)

    # log end and run times
    end_time = dt.datetime.now()
    print(f'End time: {end_time}')
    print(f'Run time: {end_time - start_time}')

    # save experiment statistics to disk
    exp_stats = {'Exp name:': EXP_NAME,
                 'Start time:': start_time,
                 'End time:': end_time,
                 'Run time:': (end_time - start_time),
                 # 'Loss (valid):': loss_valid,
                 '': ''}  # TODO : add more statistics to the log file.

    exp_stats_path = pj(exp_dir, f'{EXP_NAME}.log')
    dump(exp_stats.items(), exp_stats_path, append=True)
