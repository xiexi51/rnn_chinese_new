import tensorflow as tf
import tensorflow.keras as keras

def exp_safe(x):
    return tf.clip_by_value(tf.exp(x), clip_value_min=1e-10, clip_value_max=1e10)
   # return tf.exp(x)

class PostProcess(keras.layers.Layer):
    def __init__(self, M, **kwargs):
        super(PostProcess, self).__init__(**kwargs)
        self.M = M
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.Wgmm = self.add_weight(shape=(input_dim, self.M * 5), initializer='glorot_uniform', name='Wgmm')
        self.bgmm = self.add_weight(shape=(self.M * 5), initializer='zeros', name='bgmm')
        self.Wsoftmax = self.add_weight(shape=(input_dim, 3), initializer='glorot_uniform', name='Wsoftmax')
        self.bsoftmax = self.add_weight(shape=(3), initializer='zeros', name='bsoftmax')

        self.built = True
    def call(self, inputs, **kwargs):
        tf.debugging.assert_all_finite(inputs, 'postprocess inputs ill')

        R5M = tf.matmul(inputs, self.Wgmm) + self.bgmm
        _pi = R5M[:, :, :self.M]
        pi = exp_safe(_pi) / tf.reduce_sum(exp_safe(_pi), axis=-1, keepdims=True)
        mux = R5M[:, :, self.M:self.M * 2]
        muy = R5M[:, :, self.M * 2:self.M * 3]
        sigmax = exp_safe(R5M[:, :, self.M * 3:self.M * 4])
        sigmay = exp_safe(R5M[:, :, self.M * 4:])
        R3 = tf.matmul(inputs, self.Wsoftmax) + self.bsoftmax
        p = exp_safe(R3) / tf.reduce_sum(exp_safe(R3), axis=-1, keepdims=True)

        return tf.concat([pi, mux, muy, sigmax, sigmay, p], axis=-1)
    def get_config(self):
        config = super(PostProcess, self).get_config()
        config.update({"M": self.M})
        return config