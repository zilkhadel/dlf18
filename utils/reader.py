import random
import shutil
import datetime as dt
from glob import glob

from keras.preprocessing.image import ImageDataGenerator

from utils.shortcuts import pj, pe, ps, mkdirs, dump, Paths


M_SUBJECTS = ['Romain', 'Quentin', 'Pierre', 'Olivier', 'Nicolas', 'Laurent', 'Julien', 'Jerome', 'Clement', 'Cedric', 'Benjamin', 'Arthur', 'Alexis']
F_SUBJECTS = ['Veronique', 'Mathilde', 'Jeanne', 'Emma', 'Emilie', 'Claire', 'Charlotte', 'Celine', 'Caroline', 'Barbara', 'Aurore', 'Aurelie', 'Anna', 'Angelique', 'Amandine']
SUBJECTS = {'M': M_SUBJECTS, 'F': F_SUBJECTS}


def get_data_generator(data_dir, img_size, batch_size, shuffle=False, rotation_range=0., width_shift_range=0., height_shift_range=0., shear_range=0., zoom_range=0., horizontal_flip=False, fill_mode='nearest'):

    # generate an ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )

    # read samples from data_dir
    data_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=shuffle,
    )

    return data_generator


def get_train_and_valid_generators(exp_data_dir, batch_size, image_size):

    # set train and validation dirs
    train_dir = pj(exp_data_dir, 'train')
    validation_dir = pj(exp_data_dir, 'validation')

    # init train data generator
    train_generator = get_data_generator(train_dir, image_size, batch_size, shuffle=True)  # rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True

    # init validataion data generator
    validation_generator = get_data_generator(validation_dir, image_size, batch_size, shuffle=False)

    return train_generator, validation_generator


def gen_exp_data_dir(gender, train_samples, validation_samples, subjects=None):
    """
    Generates a data dir for the experiment, using either random or specified subjects of a given gender with the specified train/validation sets sizes.
    :param gender: the gender of the subjects to train model on
    :param train_samples: the number of images to use in training phase
    :param validation_samples: the number of images to use in validation phase
    :param subjects: [optional] a tuple of exactly two subject names, of the same gender.
    :return: the experiment name, the experiment data dir, actual number of training samples, actual number of validation samples
    """

    gender_source_dir = pj(Paths.data_dir, gender)

    # use specified subjects
    if subjects:
        assert len(subjects) == 2, f'Invalid size ({len(subjects)}) for subjects argument. Must be exactly 2!'
        assert all(subject in SUBJECTS[gender] for subject in subjects), f'Both subjects must be of the specified gender: {gender}'

        subject1_source_dir = pj(gender_source_dir, subjects[0])
        subject2_source_dir = pj(gender_source_dir, subjects[1])
        subject1_name = subjects[0]
        subject2_name = subjects[1]

        assert pe(subject1_source_dir) and pe(subject2_source_dir), f'Images dir for either {subjects[0]} or {subjects[1]} was not found!'

    # get two random subjects from relevant gender dir
    else:
        gender_source_subjects = glob(pj(gender_source_dir, '*'))
        random.shuffle(gender_source_subjects)
        subject1_source_dir = gender_source_subjects[0]
        subject2_source_dir = gender_source_subjects[1]
        subject1_name = ps(subject1_source_dir)[1]
        subject2_name = ps(subject2_source_dir)[1]

    # get images for both subjects
    subject1_image_paths = glob(pj(subject1_source_dir, '*.jpg'))
    subject2_image_paths = glob(pj(subject2_source_dir, '*.jpg'))

    # get the minimum number of images between subject 1 and 2 (to create train/validation sets of same sizes)
    min_num_images = min(len(subject1_image_paths), len(subject2_image_paths))

    # if there are not enough images for requested train and validation samples, use same ratio of train/validation with available images
    if train_samples + validation_samples > min_num_images:
        train_validation_ratio = train_samples / (train_samples + validation_samples)
        train_samples = int(min_num_images * train_validation_ratio)
        validation_samples = min_num_images - train_samples

    assert train_samples and validation_samples, 'Train and Validation sets must be larger than 0'

    # print sets sizes
    print(f'Using {train_samples} training samples and {validation_samples} validation samples.')

    # get random training and validation images for subject 1
    random.shuffle(subject1_image_paths)
    subject1_train_image_paths = subject1_image_paths[:train_samples]
    subject1_validation_image_paths = subject1_image_paths[train_samples:train_samples+validation_samples]

    # get random training and validation images for subject 2
    random.shuffle(subject2_image_paths)
    subject2_train_image_paths = subject2_image_paths[:train_samples]
    subject2_validation_image_paths = subject2_image_paths[train_samples:train_samples+validation_samples]

    # init experiment data dir with train and validation data dirs
    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H%M')
    exp_name = f'{subject1_name}_{subject2_name}_{timestamp}'
    exp_data_dir = pj(Paths.experiments_dir, exp_name)

    # init train data dirs
    train_dir = pj(exp_data_dir, 'train')
    subject1_train_dir = pj(train_dir, subject1_name)
    subject2_train_dir = pj(train_dir, subject2_name)
    mkdirs(subject1_train_dir)
    mkdirs(subject2_train_dir)

    # init validation data dirs
    validation_dir = pj(exp_data_dir, 'validation')
    subject1_validation_dir = pj(validation_dir, subject1_name)
    subject2_validation_dir = pj(validation_dir, subject2_name)
    mkdirs(subject1_validation_dir)
    mkdirs(subject2_validation_dir)

    # copy training and validation images for subject 1 and 2 to exp_data_dir
    for ip in subject1_train_image_paths:
        shutil.copy(ip, subject1_train_dir)

    for ip in subject2_train_image_paths:
        shutil.copy(ip, subject2_train_dir)

    for ip in subject1_validation_image_paths:
        shutil.copy(ip, subject1_validation_dir)

    for ip in subject2_validation_image_paths:
        shutil.copy(ip, subject2_validation_dir)

    # write experiment metadata
    metadata = {'Gender': gender, 'Training samples': train_samples, 'Validation samples': validation_samples}
    dump(metadata.items(), pj(exp_data_dir, 'metadata.txt'))
    dump(subject1_train_image_paths + subject2_train_image_paths, pj(exp_data_dir, 'train_paths.txt'))
    dump(subject1_validation_image_paths + subject2_validation_image_paths, pj(exp_data_dir, 'validation_paths.txt'))

    return exp_name, exp_data_dir, train_samples, validation_samples
