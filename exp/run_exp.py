import datetime as dt

from cnn_finetune.vgg16 import vgg16_model

from utils.shortcuts import pj, dump, Paths
from utils.reader import gen_exp_data_dir, get_train_and_valid_generators
from utils.saver import WeightsSaver
from utils.metrics import get_f1


def run_experiemnt(gender, train_samples, validation_samples):

    start_time = dt.datetime.now()
    print(f'Start time: {start_time}')

    # example parameters:
    # gender = 'M'                    # the gender of the subjects for which to create an experiment
    # train_samples = 70              # number of training samples
    # validation_samples = 30         # number of validation samples

    img_rows, img_cols = 224, 224   # height and width of input images
    img_channels = 3                # number of color channels in input images
    num_classes = 2                 # number of ouput classes
    batch_size = 16                 # number of training samples per gradient update
    epochs = 10                     # number of iteration over the entire training set
    steps_per_epoch = 5             # number of steps (batches of samples) to perform before declaring one epoch finished and starting the next epoch (typically equals train_samples / batch_size).
    valid_steps_per_epoch = 2       # number of steps (batches of samples) to perform before declaring one epoch finished and starting the next epoch (typically equals validation_samples / batch_size).
    freeze_first_layers = 36        # number of first layers to freeze
    save_each = 5                   # number of batches after which to save weights
    learning_rate = 0.001           # the step to use in each gradient update

    # generate data for experiment
    exp_name, exp_data_dir, actual_train_samples, actual_valid_samples = gen_exp_data_dir(gender, train_samples, validation_samples)

    # get training and validation data generators
    train_generator, validation_generator = get_train_and_valid_generators(exp_data_dir, batch_size, img_rows)

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
                        callbacks=[WeightsSaver(model, save_each, exp_data_dir)])

    # save final model weights to disk
    model.save_weights(pj(exp_data_dir, f'{exp_name}_weights_final.h5'))

    # make predictions on the validation/test set
    validation_predictions = model.predict_generator(validation_generator, verbose=1)

    # cross-entropy loss score on the validation/test set
    # loss_valid = log_loss(validation_generator, validation_predictions)

    # save model predictions to disk
    validation_predictions_path = pj(exp_data_dir, f'{exp_name}_validations_predictions.csv')
    dump(validation_predictions, validation_predictions_path)

    # log end and run times
    end_time = dt.datetime.now()
    print(f'End time: {end_time}')
    print(f'Run time: {end_time - start_time}')

    # save experiment statistics to disk
    exp_stats = {'Exp name:': exp_name,
                 'Start time:': start_time,
                 'End time:': end_time,
                 'Run time:': (end_time - start_time),
                 # 'Loss (valid):': loss_valid,
                 '': ''}  # TODO : add more statistics to the log file.

    exp_stats_path = pj(exp_data_dir, f'{exp_name}.log')
    dump(exp_stats.items(), exp_stats_path, append=True)
