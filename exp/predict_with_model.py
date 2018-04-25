import numpy as np

from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, classification_report
from cnn_finetune.vgg16 import vgg16_model, IMG_SIZE as vgg16_img_size
from cnn_finetune.small_convnet import binary_convnet_model, IMG_SIZE as bcn_img_size

from utils.shortcuts import pj, ps, mkdirs, dump, objdump
from utils.reader import get_data_generator


def predict_with_model(data_dir, out_dir, model_name, initial_weights_path, num_classes, batch_size=16):

    # initial_weights_path = pj(Paths.pretrained_dir, 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    # initial_weights_num_classes = 1000
    initial_weights = (initial_weights_path, num_classes)

    # create output dir
    mkdirs(out_dir)

    # load model
    img_size = None
    if model_name == 'vgg16':
        model = vgg16_model(num_classes=num_classes, initial_weights=initial_weights)
        img_size = vgg16_img_size
    elif model_name == 'bcn':
        model = binary_convnet_model(initial_weights_path=initial_weights_path)
        img_size = bcn_img_size
    else:
        model = None

    assert model, f'model {model_name} not loaded!'
    print(f'Loading {model_name} model | Initial weights path: {initial_weights_path} | Initial weights number of classes: {num_classes}')

    # init data generator
    data_generator = get_data_generator(data_dir, img_size, batch_size)

    # make predictions
    data_predictions = model.predict_generator(data_generator, verbose=1)

    # convert the probabilities matrix to an array of predicted classes
    if num_classes > 2:
        data_y_pred = np.array([np.argmax(p) for p in data_predictions], dtype=np.float32)

    else:  # if binary classification
        data_y_pred = np.array([0 if p > 0.5 else 1 for p in data_predictions], dtype=np.float32)

    # generate an array of true classes. Important: data generator must be used with shuffle=False for this to work.
    data_y_true = data_generator.classes  # np.array([[1-yt, yt] for yt in data_generator.classes], dtype=np.float32)

    # generate predictions object and save it to out_dir
    exp_name = 'exp'  # TODO : construct exp name
    predictions_data_path = pj(out_dir, f'{exp_name}_predictions.csv')
    print(f'Saving predictions on data set | Path: {predictions_data_path}')
    predictions_data = [('class', 'filename', 'y_true', 'y_pred', 'prob_0')]
    predictions_data += [(data_generator.class_indices[ps(data_generator.filenames[ind])[0]],
                          ps(data_generator.filenames[ind])[1],
                          data_y_true[ind],
                          int(data_y_pred[ind]),
                          pred[0]) for ind, pred in enumerate(data_predictions)]
    dump(predictions_data, predictions_data_path, delimiter=',')
    objdump([data_y_true, data_y_pred], pj(out_dir, f'{exp_name}_predictions.pkl'))

    # cross-entropy loss score on the data/test set
    print('Getting metrics on data set predictions')
    data_loss = log_loss(data_y_true, data_y_pred)
    data_accuracy = accuracy_score(data_y_true, data_y_pred)
    data_precision = precision_score(data_y_true, data_y_pred, average='micro')
    data_recall = recall_score(data_y_true, data_y_pred, average='micro')
    data_f1 = f1_score(data_y_true, data_y_pred, average='micro')

    # save classification report
    report = classification_report(data_y_true, data_y_pred, list(data_generator.class_indices.values()), list(data_generator.class_indices.keys()))
    print(report)
    dump(report, pj(out_dir, 'report.txt'))

    # save experiment statistics to disk
    exp_stats = {'Exp name:': exp_name,
                 # 'Gender:': gender,
                 # 'Start time:': start_time,
                 # 'End time:': end_time,
                 # 'Run time:': (end_time - start_time),
                 '-': '-',
                 'Data dir:': data_dir,
                 'Model name:': model_name,
                 'Initial weights path:': initial_weights_path,
                 'Num. classess:': num_classes,
                 '--': '--',
                 # 'Validation samples:': total_data_samples,
                 'Validation loss:': data_loss,
                 'Validation accuracy:': data_accuracy,
                 'Validation precision:': data_precision,
                 'Validation recall:': data_recall,
                 'Validation F1:': data_f1,
                 '': ''}

    exp_stats_path = pj(out_dir, f'{exp_name}.log')
    dump(exp_stats.items(), exp_stats_path, append=True)
