import keras.backend as K

def percision(y_true, y_pred):

    # count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))

    # if there are no true samples, fix the F1 score at 0.
    if c2 == 0:
        return 0

    # how many selected items are relevant?
    precision_score = c1 / c2

    return precision_score

def recall(y_true, y_pred):

    # count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # if there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # how many relevant items are selected?
    recall_score = c1 / c3

    return recall_score

def f1(y_true, y_pred):

    precision_s = percision(y_true, y_pred)

    recall_s = recall(y_true, y_pred)

    # calculate F1 score
    f1_score = 2 * (precision_s * recall_s) / (precision_s + recall_s)
    return f1_score