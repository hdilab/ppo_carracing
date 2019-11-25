import logging
logging.basicConfig(format='%(asctime)s %(message)s', filename='custom/train_ppo.log', level=logging.INFO)
import gym

import math
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from hyperparameters import *
import cv2
from models.ppo import PPO
from buffer import Buffer
import time
import json
from collections import deque
import os
import datetime
import horovod.tensorflow as hvd

from tensorflow.keras import layers

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')

print("Gpus", gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    print("Setting gpu", gpus[hvd.local_rank()])
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank() % len(gpus)], 'GPU')

if hvd.local_rank() == 0:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'custom/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    train_summary_writer.set_as_default()


def compute_gae(rewards, values, bootstrap_values, dones, gamma, lam):
    values = np.vstack((values, [bootstrap_values]))
    deltas = []
    for i in reversed(range(len(rewards))):
        V = rewards[i] + (1.0 - dones[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))

    A = deltas[-1, :]
    advantages = [A]
    for i in reversed(range(len(deltas) - 1)):
        A = deltas[i] + (1.0 - dones[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)
    advantages = np.array(list(advantages))
    return advantages


def preprocess_frame(frame):
    frame = frame[0:84, :, :]
    frame = cv2.resize(frame, (64, 64))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = np.asarray(frame, np.float32) / 255
    frame = np.reshape(frame, (64, 64, 1))
    return frame


def update(ppo, states, actions, returns, advantages, old_log_prob, epoch, timesteps):

    tf.summary.scalar('advantages', np.mean(advantages), step=timesteps)
    tf.summary.scalar('returns', np.mean(returns), step=timesteps)
    inds = np.arange(states.shape[0])
    for i in range(TRAIN_K_MINIBATCH):
        np.random.shuffle(inds)
        for start in range(0, STEPS_PER_EPOCH, STEPS_PER_BATCH):
            end = start + STEPS_PER_BATCH
            mbinds = inds[start:end]
            slices = (arr[mbinds] for arr in (states, actions, returns, advantages, old_log_prob))
            pi_loss, value_loss, entropy_loss, total_loss, old_neg_log_val, neg_log_val, approx_kl, ratio = \
                ppo.loss(*slices)
            if hvd.rank() == 0:
                tf.summary.scalar('pi_loss', np.asscalar(pi_loss.numpy()), step=timesteps)
                tf.summary.scalar('value_loss', np.asscalar(value_loss.numpy()), step=timesteps)
                tf.summary.scalar('old_neg_log_val', np.asscalar(old_neg_log_val.numpy()), step=timesteps)
                tf.summary.scalar('neg_log_val', np.asscalar(neg_log_val.numpy()), step=timesteps)
                tf.summary.scalar('approx_kl', np.asscalar(approx_kl.numpy()), step=timesteps)
                tf.summary.scalar('entropy_loss', np.asscalar(entropy_loss.numpy()), step=timesteps)
                tf.summary.scalar('total_loss', np.asscalar(total_loss.numpy()), step=timesteps)
                tf.summary.scalar('ratio', np.mean(ratio), step=timesteps)
            timesteps += STEPS_PER_BATCH
        if epoch == 0 and i == 0:
            hvd.broadcast_variables(ppo.variables, root_rank=0)
            hvd.broadcast_variables(ppo.optimizer.variables(), root_rank=0)
    return timesteps


env = gym.make("CarRacing-v0")
stacked_frames = deque(maxlen=4)
# stacked_scores = deque(maxlen=10)
try:
    ppo = PPO(ACTION_SIZE, EPSILON, ENTROPY_REG, VALUE_COEFFICIENT, "CNN", LEARNING_RATE, MAX_GRAD_NORM)
    finished_games = 0
    steps = 0
    total_reward = 0
    timesteps = 0
    frame = env.reset()
    state_ = preprocess_frame(frame)
    stacked_frames.append(state_)
    stacked_frames.append(state_)
    stacked_frames.append(state_)
    stacked_frames.append(state_)
    for epoch in range(EPOCHS):
        states, actions, values, rewards, dones, old_log_pi = [], [], [], [], [], []

        for t in range(int(STEPS_PER_EPOCH)):
            stacked_states = np.concatenate(stacked_frames, axis=2)

            pi, old_log_p, v = ppo.call(np.expand_dims(stacked_states, axis=0))
            pi = pi.numpy()[0]
            clipped_actions = np.clip(pi, env.action_space.low, env.action_space.high)
            frame, reward, done, _ = env.step(clipped_actions)
            total_reward += reward

            states.append(stacked_states)
            actions.append(pi)
            values.append(v.numpy()[0])
            rewards.append(reward)
            dones.append(done)
            old_log_pi.append(old_log_p.numpy()[0])

            if done:
                frame = env.reset()
                state_ = preprocess_frame(frame)
                for _ in range(4):
                    stacked_frames.append(state_)
                finished_games += 1
                if hvd.rank() == 0:
                    tf.summary.scalar('episode reward', total_reward, step=finished_games)
                total_reward = 0
            else:
                state_ = preprocess_frame(frame)
                stacked_frames.append(state_)

        stacked_states = np.concatenate(stacked_frames, axis=2)
        pi, old_log_p, v = ppo.call(np.expand_dims(stacked_states, axis=0))
        last_val = v.numpy()[0]

        advantages = compute_gae(
            rewards, values, last_val, dones, GAMMA, LAMBDA)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        timesteps = update(ppo, np.array(states), np.array(actions), np.array(returns), np.array(advantages), np.array(old_log_pi), epoch, timesteps)
        if hvd.rank() == 0:
            print("Completed epoch", epoch)
            ppo.save_weights("custom/ppo.h5")
except Exception as ex:
    logging.exception(ex)
finally:
    if env is not None:
        env.close()
