import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

import gym

import os

import random

# reproducibility
SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Hyperparameters
gamma = 0.99
max_steps_per_episode = 10000
learning_rate = 0.01

# Create the CartPole environment
env = gym.make("CartPole-v1")
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

# Build the model
inputs = layers.Input(shape=(num_states,))
common = layers.Dense(128, activation="relu")(inputs)

# Actor: policy (what action to take)
action = layers.Dense(num_actions, activation="softmax")(common)
# Critic: value of the state (how good the state is)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
huber_loss = keras.losses.Huber()

# Main training loop
running_reward = 0  # 부드러운 평균 보상
episode_count = 0

while True:
    state, _ = env.reset(seed=SEED)
    episode_reward = 0

    with tf.GradientTape() as tape:
        all_rewards = []
        all_action_probs = []
        all_critic_values = []

        for timestep in range(1, max_steps_per_episode):
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)

            action_probs, critic_value = model(state_tensor)

            all_critic_values.append(critic_value[0, 0])

            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            all_action_probs.append(tf.math.log(action_probs[0, action]))

            state, reward, done, _, _ = env.step(action)
            all_rewards.append(reward)
            episode_reward += reward

            if done:
                break

        returns = []
        discounted_sum = 0
        for r in all_rewards[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)
        returns = returns.tolist()

        history = zip(all_action_probs, all_critic_values, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            advantage = ret - value
            actor_losses.append(-log_prob * advantage)
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_count += 1
    running_reward = (
        0.05 * episode_reward + (1 - 0.05) * running_reward
    )  # exponential moving average
    if episode_count % 10 == 0:
        print(f"Episode {episode_count}: running reward: {running_reward:.2f}")

    if running_reward > 195:  # 목표 보상
        print(f"Solved at episode {episode_count}!")
        model.save_weights("models/actor_critic.weights.h5")
        break
