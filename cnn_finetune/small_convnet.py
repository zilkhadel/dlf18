from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


IMG_SIZE = 224
IMG_CHANNELS = 3


def binary_convnet_model(img_rows=IMG_SIZE, img_cols=IMG_SIZE, channels=IMG_CHANNELS, initial_weights_path=None, metrics=None):

    # construct model
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # load pre-trained weights
    if initial_weights_path:
        model.load_weights(initial_weights_path)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=metrics)

    return model
