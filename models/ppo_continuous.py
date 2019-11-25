import tensorflow as tf
from tensorflow.keras import layers, models
import math
import horovod.tensorflow as hvd
import numpy as np
EPS = 1e-8


class Actor(layers.Layer):
    def __init__(self, action_size, epsilon, entropy_reg, initial_layer, learning_rate):
        super(Actor, self).__init__()
        self.epsilon = epsilon
        self.action_size = action_size
        self.entropy_reg = entropy_reg
        # self.dense = layers.Dense(40)
        self.mu = layers.Dense(action_size, activation='tanh')
        self.log_std = tf.Variable(-0.5 * tf.zeros(self.action_size, dtype=tf.float32))
        if initial_layer == "MLP":
            self.initial_layer = MLP()
        else:
            self.initial_layer = CNN()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def call(self, inputs):
        x = self.initial_layer(inputs)
        # x = self.dense(x)
        mu = self.mu(x)
        log_std = self.log_std
        std = tf.exp(log_std)
        pi = mu + tf.random.normal(tf.shape(mu)) * std
        pi = tf.clip_by_value(pi, -2.0, 2.0)
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)
        return pi, logp_pi, mu, log_std

    @tf.function
    def gaussian_likelihood(self, x, mu, log_std):
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    @tf.function
    def loss(self, x, actions, advantages, log_p_old):
        with tf.GradientTape() as tape:
            pi, logp_pi, mu, log_std = self.call(x)
            logp = self.gaussian_likelihood(actions, mu, log_std)
            ratio = tf.math.exp(logp - log_p_old)
            min_adv = tf.where(advantages > 0, (1 + self.epsilon) * advantages, (1 - self.epsilon) * advantages)
            loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))
        # tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss


class Critic(layers.Layer):
    def __init__(self, initial_layer, learning_rate):
        super(Critic, self).__init__()
        self.value = layers.Dense(1, activation=None)
        if initial_layer == "MLP":
            self.initial_layer = MLP()
        else:
            self.initial_layer = CNN()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def call(self, inputs):
        x = self.initial_layer(inputs)
        value = self.value(x)
        return value

    @tf.function
    def loss(self, x, returns):
        with tf.GradientTape() as tape:
            values = self.call(x)
            loss = tf.reduce_mean((returns - values) ** 2)
        # tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss


class CNN(layers.Layer):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = layers.Conv2D(32, 8, 4, activation='elu', kernel_initializer=tf.initializers.orthogonal(gain=math.sqrt(2)))
        self.conv_2 = layers.Conv2D(64, 4, 2, activation='elu', kernel_initializer=tf.initializers.orthogonal(gain=math.sqrt(2)))
        self.conv_3 = layers.Conv2D(64, 3, 1, activation='elu', kernel_initializer=tf.initializers.orthogonal(gain=math.sqrt(2)))
        self.flatten = layers.Flatten()

    @tf.function
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        return x


class MLP(layers.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense_1 = layers.Dense(40)
        self.dense_2 = layers.Dense(40, activation='relu')

    @tf.function
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x


class PPO(tf.keras.Model):
    def __init__(self, action_size, epsilon, entropy_reg, value_coeff, initial_layer, learning_rate):
        super(PPO, self).__init__()

        self.actor = Actor(action_size, epsilon, entropy_reg, initial_layer, learning_rate)
        self.critic = Critic(initial_layer, learning_rate)

        self.value_coeff = value_coeff

    @tf.function
    def call(self, inputs):
        pi, log_pi, mu, log_std = self.actor(inputs)
        value = self.critic(inputs)
        return pi, log_pi, value

    @tf.function
    def loss(self, states, advantages, returns, actions, log_prob_old):
        value_loss = self.critic.loss(states, returns)
        pi_loss = self.actor.loss(states, actions, advantages, log_prob_old)
        return pi_loss, value_loss








