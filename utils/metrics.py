import keras.backend as K


def get_counts(y_true, y_pred):
    # count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    return c1, c2, c3


def get_percision(y_true, y_pred):
    # count positive samples.
    c1, c2, c3 = get_counts(y_true, y_pred)

    # if there are no true samples, fix the F1 score at 0.
    if c2 == 0:
        return 0

    # how many selected items are relevant?
    precision = c1 / c2

    return precision


def get_recall(y_true, y_pred):
    # count positive samples.
    c1, c2, c3 = get_counts(y_true, y_pred)

    # if there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # how many relevant items are selected?
    recall = c1 / c3

    return recall


def get_f1(y_true, y_pred):

    precision = get_percision(y_true, y_pred)

    recall = get_recall(y_true, y_pred)

    # calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1
