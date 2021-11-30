import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import AbstractRNNCell
from tensorflow.python.util import nest

def exp_safe(x):
    return tf.clip_by_value(tf.exp(x), clip_value_min=1e-10, clip_value_max=1e10)

class SGRUCell(AbstractRNNCell, keras.layers.Layer):
    def __init__(self, units, nclass, **kwargs):
        super(SGRUCell, self).__init__(**kwargs)
        self.units = units
        self.nclass = nclass

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        self.bias_z = self.add_weight(shape=(self.units), initializer=keras.initializers.constant(5), name='bias_z')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), initializer='orthogonal', name='recurrent_kernel')
        self.kernel_c = self.add_weight(shape=(self.nclass, self.units * 4), initializer='glorot_uniform', name='kernel_c')
        self.bias = self.add_weight(shape=(self.units * 3), initializer='zeros', name='bias')
        self.built = True

    def call(self, inputs, states, training):
        tf.debugging.assert_all_finite(inputs, 'sgrucell inputs ill')
        h_tm1 = states[0] if nest.is_sequence(states) else states  # previous memory
        ch = tf.cast(inputs[:, 0], tf.int32)
        _ch = tf.one_hot(ch, self.nclass)
        z = tf.sigmoid(tf.matmul(h_tm1, self.recurrent_kernel[:, :self.units])
                       + tf.matmul(_ch, self.kernel_c[:, :self.units])
                       + self.bias_z)
        r = tf.sigmoid(tf.matmul(h_tm1, self.recurrent_kernel[:, self.units:self.units * 2])
                       + tf.matmul(_ch, self.kernel_c[:, self.units:self.units * 2])
                       + self.bias[:self.units])
        hh = tf.tanh(tf.matmul(r * h_tm1, self.recurrent_kernel[:, self.units * 2:self.units * 3])
                       + tf.matmul(_ch, self.kernel_c[:, self.units * 2:self.units * 3])
                       + self.bias[self.units * 1:self.units * 2])
        h = z * h_tm1 + (1 - z) * hh
        o = tf.tanh(tf.matmul(h, self.recurrent_kernel[:, self.units * 3:])
                       + tf.matmul(_ch, self.kernel_c[:, self.units * 3:])
                       + self.bias[self.units * 2:])
        new_state = [h] if nest.is_sequence(states) else h
        return o, new_state

    def get_config(self):
        config = super(SGRUCell, self).get_config()
        config.update({'units': self.units, 'nclass': self.nclass})
        return config