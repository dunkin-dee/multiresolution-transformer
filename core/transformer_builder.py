"""Custom Keras layers and the learning-rate schedule.

Note on ``package="scalper"``: this string is the *serialization key* Keras writes
into saved ``.keras`` files, which are then looked up as ``"scalper>LayerName"``
on load. It is deliberately kept at the project's original name even though the
project has been renamed — changing it silently breaks ``load_model()`` for every
checkpoint saved before the change. It is not a display name; leave it alone.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, Layer, LayerNormalization


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(maxlen, d_model)

    def positional_encoding(self, maxlen, d_model):
        positions = np.arange(maxlen)[:, np.newaxis]
        angles = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (angles // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


@tf.keras.utils.register_keras_serializable(package="scalper")
class LearnablePositionalEncoding(Layer):
    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(self.max_seq_len, self.embed_dim),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self.pos_embedding[:seq_len, :]
        return inputs + pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_seq_len': self.max_seq_len,
            'embed_dim': self.embed_dim
        })
        return config


@tf.keras.utils.register_keras_serializable(package="scalper")
class StochasticGatedTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,
                 attention_dropout=0.1, stochastic_depth_rate=0.1, **kwargs):
        super(StochasticGatedTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.attention_dropout = attention_dropout
        self.stochastic_depth_rate = stochastic_depth_rate

    def build(self, input_shape):
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim//self.num_heads,
            dropout=self.attention_dropout
        )
        self.gate_att = tf.keras.layers.Dense(self.embed_dim, activation='sigmoid')
        self.ff1 = tf.keras.layers.Dense(self.ff_dim, activation="gelu")
        self.ff2 = tf.keras.layers.Dense(self.embed_dim)
        self.gate_ffn = tf.keras.layers.Dense(self.embed_dim, activation='sigmoid')
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.rate)
        self.dropout2 = Dropout(self.rate)
        self.att.build(query_shape=input_shape, value_shape=input_shape)
        self.gate_att.build(input_shape)
        self.ff1.build(input_shape)
        self.ff2.build((input_shape[0], input_shape[1], self.ff_dim))
        self.gate_ffn.build(input_shape)
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        super().build(input_shape)

    def stochastic_depth(self, x, training):
        if not isinstance(training, bool):
            training = tf.cast(training, tf.bool)
        keep_prob = 1.0 - self.stochastic_depth_rate

        def drop_path():
            batch_size = tf.shape(x)[0]
            random_tensor = keep_prob
            random_tensor += tf.random.uniform([batch_size, 1, 1], dtype=x.dtype)
            binary_tensor = tf.floor(random_tensor)
            return x * binary_tensor / keep_prob

        output = tf.cond(
            tf.logical_and(training, tf.constant(self.stochastic_depth_rate > 0)),
            lambda: drop_path(),
            lambda: x
        )
        return output

    def call(self, inputs, training=True):
        if isinstance(training, str):
            training = training == 'True'

        def skip_block():
            return inputs

        def process_block():
            def apply_training_noise():
                noise_scale = 0.01
                input_noise = tf.random.normal(tf.shape(inputs), stddev=noise_scale, dtype=inputs.dtype)
                return inputs + input_noise

            inputs_with_noise = tf.cond(
                tf.cast(training, tf.bool),
                lambda: apply_training_noise(),
                lambda: inputs
            )

            attn_output = self.att(
                query=inputs_with_noise,
                key=inputs_with_noise,
                value=inputs_with_noise
            )

            gate_val = self.gate_att(inputs)

            def add_gate_noise():
                gate_noise = tf.random.normal(tf.shape(gate_val), mean=1.0, stddev=0.1, dtype=gate_val.dtype)
                return gate_val * gate_noise

            gate_val = tf.cond(
                tf.cast(training, tf.bool),
                lambda: add_gate_noise(),
                lambda: gate_val
            )

            attn_output = gate_val * attn_output
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)

            ffn_output = self.ff1(out1)

            def apply_feature_dropout():
                feature_mask = tf.cast(
                    tf.random.uniform(tf.shape(ffn_output)) > 0.1,
                    dtype=ffn_output.dtype
                )
                return ffn_output * feature_mask * tf.cast(1.1, dtype=ffn_output.dtype)

            ffn_output = tf.cond(
                tf.cast(training, tf.bool),
                lambda: apply_feature_dropout(),
                lambda: ffn_output
            )

            ffn_output = self.ff2(ffn_output)
            gate_val = self.gate_ffn(out1)

            def add_ffn_gate_noise():
                gate_noise = tf.random.normal(tf.shape(gate_val), mean=1.0, stddev=0.1, dtype=gate_val.dtype)
                return gate_val * gate_noise

            gate_val = tf.cond(
                tf.cast(training, tf.bool),
                lambda: add_ffn_gate_noise(),
                lambda: gate_val
            )

            ffn_output = gate_val * ffn_output
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

        random_value = tf.random.uniform([], dtype=tf.float32)
        skip_condition = tf.logical_and(
            tf.cast(training, tf.bool),
            tf.less(random_value, tf.constant(self.stochastic_depth_rate))
        )
        return tf.cond(skip_condition, skip_block, process_block)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'attention_dropout': self.attention_dropout,
            'stochastic_depth_rate': self.stochastic_depth_rate
        })
        return config


@tf.keras.utils.register_keras_serializable(package="scalper")
class AddTypeEmbedding(Layer):
    def __init__(self, type_id, embed_dim=16, num_types=3, **kwargs):
        super().__init__(**kwargs)
        self.type_id = type_id
        self.embed_dim = embed_dim
        self.num_types = num_types
        self.embedding_layer = None

    def build(self, input_shape):
        self.embedding_layer = Embedding(
            input_dim=self.num_types,
            output_dim=self.embed_dim,
            name=f'{self.name}_embed'
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        type_ids = tf.fill((batch_size, seq_len), self.type_id)
        type_embed = self.embedding_layer(type_ids)
        return tf.concat([inputs, type_embed], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "type_id": self.type_id,
            "embed_dim": self.embed_dim,
            "num_types": self.num_types
        })
        return config


@tf.keras.utils.register_keras_serializable(package="scalper")
class AttentionPooling(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = Dense(1)
        super().build(input_shape)

    def call(self, inputs):
        attention_logits = self.attention_dense(inputs)
        attention_weights = tf.nn.softmax(attention_logits, axis=1)
        return tf.reduce_sum(inputs * attention_weights, axis=1)

    def get_config(self):
        config = super().get_config()
        return config


@tf.keras.utils.register_keras_serializable(package="scalper")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=5e-5, warmup_steps=90000, decay_steps=900000):
        super(WarmupCosineDecay, self).__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def __call__(self, step):
        warmup_lr = self.initial_lr * (tf.cast(step, tf.float32) /
                                       tf.cast(self.warmup_steps, tf.float32))
        cosine_decay = 0.5 * (1 + tf.cos(
            3.14159 * (tf.cast(step, tf.float32) - self.warmup_steps) /
            tf.cast(self.decay_steps - self.warmup_steps, tf.float32)))
        decay_lr = self.initial_lr * cosine_decay
        lr = tf.where(step < self.warmup_steps, warmup_lr, decay_lr)
        return lr

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps
        }
