from keras.callbacks import Callback
from utils.shortcuts import pj, dump


class WeightsSaver(Callback):
    def __init__(self, model, save_each, exp_dir):
        Callback.__init__(self)
        self.model = model
        self.save_each = save_each
        self.exp_dir = exp_dir
        self.batch = 0

    def on_batch_end(self, batch, logs=None):
        overwrite = True  # if set to False, will keep weights from previous batches (uses a lot of storage!)
        if self.batch % self.save_each == 0:
            weights_path = pj(self.exp_dir, 'weights.h5' if overwrite else f'weights_{self.batch}.h5')
            log_path = weights_path.replace('.h5', '.log')
            self.model.save_weights(weights_path)
            log = {**logs, **self.params}
            dump(log.items(), log_path)
        self.batch += 1
