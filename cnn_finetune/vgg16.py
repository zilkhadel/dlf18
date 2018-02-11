from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Convolution2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten  # , Input, AveragePooling2D, merge, Reshape, Activation


def vgg16_model(img_rows, img_cols, channels=1, num_classes=None, initial_weights=None, freeze_first_layers=None, learning_rate=1e-3, metrics=None):
    """VGG 16 Model for Keras

    Model Schema is based on
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

    ImageNet Pretrained Weights
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channels - 1 for grayscale, 3 for color
      num_classes - number of categories for our classification task
      initial_weights - a tuple of (path_to_initial_weights, initial_weights_num_classes)
      freeze_first_layers - number of layers to freeze
      learning_rate - the step to use in each gradient update
      metrics - a list of strings and/or callback functions to use for metrics at the end of each batch.
    """

    initial_weights_path = initial_weights[0] if initial_weights else None
    initial_weights_num_classes = initial_weights[1] if initial_weights else 1000

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channels)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # add output layer with initial_weights_num_classes (default: 1000, for ImageNet pretrained weights)
    model.add(Dense(initial_weights_num_classes, activation='softmax'))

    # load pre-trained weights
    if initial_weights_path:
        model.load_weights(initial_weights_path)
    else:
        raise Exception('initial_weights_path missing!')

    # replace softmax layer for transfer learning, if needed
    if num_classes != initial_weights_num_classes:
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.add(Dense(num_classes, activation='softmax'))

    # freeze layers
    for layer in model.layers[:freeze_first_layers]:
        layer.trainable = False

    # compile model
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=metrics or [])

    return model
