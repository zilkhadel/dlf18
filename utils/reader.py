from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 224  # height/width of the resized input images


def get_train_and_valid_generators(train_dir, validation_dir, batch_size):
    # generate an ImageDataGenerator that does data augmentation by applying manipulations to train samples.
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # read train samples from train_dir
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    # generate an ImageDataGenerator that does no manipulations to validation samples.
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # read validation samples from validation_dir
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator
