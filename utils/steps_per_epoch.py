def spe(train_samples, num_classes, batch_size):
    int = (train_samples * num_classes) // batch_size
    mod = (train_samples * num_classes) % batch_size
    if mod == 0:
        steps = int
    else:
        steps = int+1
    return steps

