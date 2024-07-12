import math
import tensorflow as tf
from keras import backend
from keras.saving import register_keras_serializable
from keras.src.optimizers.schedules.learning_rate_schedule import LearningRateSchedule


@register_keras_serializable(package='Miss')
class WarmHoldCoolAnnihilateScheduler(LearningRateSchedule):
    def __init__(self, min_lr, max_lr, warm_steps, hold_steps, cool_steps, annih_steps,
                 annih_factor=0.01, name=None):
        super(WarmHoldCoolAnnihilateScheduler, self).__init__()

        if not (0.0 < min_lr < max_lr):
            raise ValueError('Wrong values provided for "min_lr" and "max_lr".')
        if warm_steps < 0:
            raise ValueError('Wrong value provided for "warm_steps".')
        if hold_steps < 0:
            raise ValueError('Wrong value provided for "hold_steps".')
        if annih_steps < 0:
            raise ValueError('Wrong value provided for "annih_steps".')
        if not 0.0 < annih_factor <= 1.0:
            raise ValueError('Wrong value provided for "annih_factor".')

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warm_steps = warm_steps
        self.hold_steps = hold_steps
        self.cool_steps = cool_steps
        self.annih_steps = annih_steps
        self.annih_factor = annih_factor
        self.name = name

    def __call__(self, step):
        step = tf.cast(tf.convert_to_tensor(step), 'float32')
        min_lr = tf.convert_to_tensor(self.min_lr, 'float32')
        max_lr = tf.convert_to_tensor(self.max_lr, 'float32')
        warm_stop = float(self.warm_steps)
        hold_stop = self.hold_steps + warm_stop
        cool_stop = self.cool_steps + hold_stop
        annih_stop = self.annih_steps + cool_stop

        pred_fn_pairs = [
            # Warm up from min_lr to max_lr
            (step <= warm_stop,
             lambda: min_lr + (max_lr - min_lr) * step / warm_stop),

            # Hold on max_lr
            ((warm_stop < step) & (step <= hold_stop),
             lambda: max_lr),

            # Cool down from max_lr to min_lr
            ((hold_stop < step) & (step <= cool_stop),
             lambda: self.cool_down(min_lr, max_lr, step - hold_stop, cool_stop - hold_stop)),

            # Annihilate from min_lr to min_lr * annih_factor
            ((cool_stop < step) & (step <= annih_stop),
             lambda: min_lr * (1 - (1 - self.annih_factor) * (step - cool_stop) / (annih_stop - cool_stop))),
        ]

        # The default isn't needed here because our conditions are mutually
        # exclusive and exhaustive, but tf.case requires it.
        def _default():
            return min_lr * self.annih_factor

        curr_lr = tf.case(pred_fn_pairs, _default, exclusive=True)
        tf.summary.scalar('learning_rate', data=curr_lr, step=tf.cast(step, 'int64'))

        return curr_lr

    def cool_down(self, min_lr, max_lr, step, total):
        return max_lr - (max_lr - min_lr) * step / total

    def get_config(self):
        return {
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'warm_steps': self.warm_steps,
            'hold_steps': self.hold_steps,
            'cool_steps': self.cool_steps,
            'annih_steps': self.annih_steps,
            'annih_factor': self.annih_factor,
            'name': self.name
        }


@register_keras_serializable(package='Miss')
class WarmHoldCosineCoolAnnihilateScheduler(WarmHoldCoolAnnihilateScheduler):
    def __init__(self, min_lr, max_lr, warm_steps, hold_steps, cool_steps, cosine_cycles, annih_steps,
                 cosine_width=2.0, cosine_height=0.9, annih_factor=0.01, name=None):
        super(WarmHoldCosineCoolAnnihilateScheduler, self).__init__(
            min_lr=min_lr, max_lr=max_lr, warm_steps=warm_steps, hold_steps=hold_steps, cool_steps=cool_steps,
            annih_steps=annih_steps, annih_factor=annih_factor, name=name)

        if not (0.0 < cosine_height < 1.0):
            raise ValueError('Wrong values provided for "cosine_height".')

        first_width = cool_steps / sum(cosine_width ** i for i in range(cosine_cycles))
        if first_width <= 1:
            raise ValueError('Not enough "cool_steps" for provided "cosine_cycles" and "cosine_width".')
        else:
            tf.get_logger().info('First cosine cooldown cycle will remain ~{:.1f} steps'.format(first_width))

        self.cosine_cycles = cosine_cycles
        self.cosine_width = cosine_width
        self.cosine_height = cosine_height

    def cool_down(self, min_lr, max_lr, step, total):
        alpha = min_lr / max_lr
        eps_100 = 100 * backend.epsilon()
        first_width = (1 + eps_100) * total / sum(self.cosine_width ** i for i in range(self.cosine_cycles))
        complete_fraction = step / first_width

        cycle_idx = tf.math.floor(tf.math.log(
            1.0 - complete_fraction * (1.0 - self.cosine_width)) / tf.math.log(self.cosine_width))
        current_fraction = (1.0 - self.cosine_width ** cycle_idx) / (1.0 - self.cosine_width)
        complete_fraction = (complete_fraction - current_fraction) / self.cosine_width ** cycle_idx

        height_fraction = self.cosine_height ** cycle_idx
        cosine_decayed = 0.5 * height_fraction * (1.0 + tf.math.cos(math.pi * complete_fraction))
        decayed = (1 - alpha) * cosine_decayed + alpha

        return max_lr * decayed

    def get_config(self):
        config = super().get_config()
        config.update({
            'cosine_cycles': self.cosine_cycles,
            'cosine_width': self.cosine_width,
            'cosine_height': self.cosine_height
        })

        return config
