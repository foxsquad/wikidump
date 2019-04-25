"""Custom Keras callback style for local training."""
import tensorflow as tf
from absl import logging
from absl.flags import FLAGS
from tensorflow.python.keras import callbacks as C


class Spinner(object):
    __base__ = u'↖↗↘↙'

    def __generator_fn__(self):
        counter = 0
        base = str(self.__base__)
        base_length = len(base)

        while True:
            yield base[counter]
            counter += 1

            # Avoid counter overflow for long running process
            if counter % base_length == 0:
                counter = 0

    def __init__(self):
        super().__init__()
        self.__generator__ = self.__generator_fn__()

    def __str__(self):
        return next(self.__generator__)
    __repr__ = __str__


nan = float('nan')
spinner = Spinner() if FLAGS.decorate else ''


class SimpleLogger(C.Callback):
    """A simple end-of-epoch logger."""

    def on_batch_end(self, batch, logs=None):
        if FLAGS.update_freq and batch % FLAGS.update_freq != 0:
            return
        logs = logs or {}
        loss = logs.get('loss')
        print(f' {spinner} loss {loss:.4f}\r', end='')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', nan)
        val_loss = logs.get('val_loss', nan)
        print(f'Epoch {epoch + 1} - '
              f'loss: {loss:.4f}  '
              f'val loss: {val_loss:.4f}\r')


class SaveStateCallback(C.Callback):
    def __init__(self, state_file):
        super().__init__()
        self.state_file = state_file

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.state_file)


class ModelCheckpoint(C.Callback):
    """Checkpoint manager and monitor."""

    def __init__(self, ckpt_dir, val_dataset=None,
                 save_best_only=True, max_to_keep=None,
                 load_weights_on_model_set=True):
        super(ModelCheckpoint, self).__init__()
        self.ckpt_dir = ckpt_dir
        self.save_best_only = save_best_only
        self.best = None
        self.val_dataset = val_dataset

        self.saver = None  # type: tf.train.Checkpoint
        self.manager = None  # type: tf.train.CheckpointManager
        self.max_to_keep = max_to_keep
        self.load_weights_on_model_set = load_weights_on_model_set

    def set_model(self, model):
        super().set_model(model)

        self.saver = tf.train.Checkpoint(
            model=self.model,
            # optimizer=model.optimizer
        )
        self.manager = tf.train.CheckpointManager(
            self.saver,
            directory=self.ckpt_dir,
            max_to_keep=self.max_to_keep)

        latest_checkpoint = self.manager.latest_checkpoint

        if self.load_weights_on_model_set:
            if latest_checkpoint:
                logging.info(
                    'Restoring model weights from latest checkpoint "%s"',
                    latest_checkpoint)
                try:
                    self.saver.restore(latest_checkpoint)
                except ValueError:
                    logging.error(
                        'Latest checkpoint at "%s" not compatible with '
                        'current model structure.', latest_checkpoint)
            else:
                logging.warning(
                    'Latest checkpoint not found at dir "%s", '
                    'will initialize model weights as defined in '
                    '`model_fn`.', self.ckpt_dir)

        if self.val_dataset:
            logging.info('New model set, updating best val_loss with dataset')
            self.best = model.evaluate(self.val_dataset, verbose=0)
            logging.info('Current val_loss: %.3f', self.best)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss', nan)

        if self.manager and val_loss < self.best:
            self.manager.save(epoch)
