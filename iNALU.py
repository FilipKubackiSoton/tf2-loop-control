from typing import List
import tensorflow as tf
from tensorflow_addons.utils import types
from typeguard import typechecked


class NALURegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, reg_coef=0.1):
        self.reg_coef = reg_coef

    def __call__(self, var: List[tf.Variable]) -> tf.Tensor:
        return self.reg_coef * tf.add_n(
            [
                tf.reduce_mean(tf.math.maximum(tf.math.minimum(-v, v) + 20, 0))
                for v in var
            ]
        )

    def get_config(self):
        return {"reg_coef": float(self.reg_coef)}


class NALU(tf.keras.layers.Layer):
    @typechecked
    def __init__(
        self,
        units: int,
        regularizer: types.Regularizer = NALURegularizer(reg_coef=0.05),
        clipping: float = 20,
        w_initializer: types.Initializer = tf.random_normal_initializer(
            mean=1.0, stddev=0.1, seed=None
        ),
        m_initializer: types.Initializer = tf.random_normal_initializer(
            mean=-1.0, stddev=0.1, seed=None
        ),
        g_initializer: types.Initializer = tf.random_normal_initializer(
            mean=0.0, stddev=0.1, seed=None
        ),
        *args,
        **kwargs,
    ):
        super(NALU, self).__init__(*args, **kwargs)

        self.units = units
        self.reg_fn = regularizer
        self.clipping = clipping

        self.w_initializer = w_initializer
        self.m_initializer = m_initializer
        self.g_initializer = g_initializer

        self.gate_as_vector = True
        self.force_operation = None
        self.weights_separation = True
        self.input_gate_dependance = False
        self.initializer = None

    def build(self, input_shape):
        # action variables
        self.w_hat = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.w_initializer,
            trainable=True,
            name="w",
            use_resource=False,
        )

        self.m_hat = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.m_initializer,
            trainable=True,
            name="m",
            use_resource=False,
        )

        self.w_hat_prime = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.w_initializer,
            trainable=True,
            name="w_prime",
            use_resource=False,
        )

        self.m_hat_prime = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.m_initializer,
            trainable=True,
            name="m_prime",
            use_resource=False,
        )

        # gating varaible
        self.g = self.add_weight(
            shape=(self.units,),
            initializer=self.g_initializer,
            trainable=True,
            name="g",
            use_resource=False,
        )

    @tf.function
    def get_reg_loss(self):
        var_list = [self.w_hat, self.m_hat, self.g]
        if self.weights_separation:
            var_list += [self.w_hat_prime, self.m_hat_prime]
        return self.reg_fn(var_list)

    def call(self, input):
        eps = 1e-7
        w1 = tf.math.tanh(self.w_hat) * tf.math.sigmoid(self.m_hat)
        w2 = tf.math.tanh(self.w_hat_prime) * tf.math.sigmoid(self.m_hat_prime)
        a1 = tf.matmul(input, w1)

        m1 = tf.math.exp(
            tf.minimum(
                tf.matmul(tf.math.log(tf.maximum(tf.math.abs(input), eps)), w2),
                self.clipping,
            )
        )

        # sign
        w1s = tf.math.abs(tf.reshape(w2, [-1]))
        xs = tf.concat([input] * w1.shape[1], axis=1)
        xs = tf.reshape(xs, shape=[-1, w1.shape[0] * w1.shape[1]])
        sgn = tf.sign(xs) * w1s + (1 - w1s)
        sgn = tf.reshape(sgn, shape=[-1, w1.shape[1], w1.shape[0]])
        ms = tf.math.reduce_prod(sgn, axis=2)

        self.add_loss(lambda: self.get_reg_loss())
        g1 = tf.math.sigmoid(self.g)
        return g1 * a1 + (1 - g1) * m1 * tf.clip_by_value(ms, -1, 1)

    def reinitialize(self):
        self.g.assign(self.g_initializer(self.g.shape))
        self.w_hat.assign(self.w_initializer(self.w_hat.shape))
        self.m_hat.assign(self.m_initializer(self.m_hat.shape))
        self.w_hat_prime.assign(self.w_initializer(self.w_hat_prime.shape))
        self.m_hat_prime.assign(self.m_initializer(self.m_hat_prime.shape))
