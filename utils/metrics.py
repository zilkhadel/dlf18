import keras.backend as K


def f1_score(y_true, y_pred):

    # count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # if there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # how many selected items are relevant?
    precision = c1 / c2

    # how many relevant items are selected?
    recall = c1 / c3

    # calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
