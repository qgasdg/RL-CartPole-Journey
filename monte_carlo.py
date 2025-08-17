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

# Define size of the state and action spaces
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

# Build the model
inputs = layers.Input(shape=(num_states,))
common = layers.Dense(128, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)

# Create the model and compile it
model = keras.Model(inputs=inputs, outputs=action)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
# huber_loss = keras.losses.Huber()

action_probs_history = []
rewards_history = []  # 에피 내에서
ep_reward_history = []  # 에피소드별 보상 기록
running_reward = 0
episode_count = 0

while True:
    state, _ = env.reset(seed=SEED)
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)

            action_probs = model(state_tensor, training=True)

            action = np.random.choice(num_actions, p=np.squeeze(action_probs))

            action_probs_history.append(tf.math.log(action_probs[0, action]))

            state, reward, done, _, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        ep_reward_history.append(episode_reward)
        if len(ep_reward_history) > 100:
            del ep_reward_history[:1]
        running_reward = np.mean(ep_reward_history)

        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = returns.tolist()

        history = zip(action_probs_history, returns)
        actor_losses = []
        for log_prob, ret in history:
            actor_losses.append(-log_prob * ret)

        loss_value = sum(actor_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        action_probs_history.clear()
        rewards_history.clear()

    episode_count += 1
    if episode_count % 10 == 0:
        print(f"Episode {episode_count}: running reward = {running_reward}")

    if running_reward > 195:
        print(f"Solved at episode {episode_count}!")
        model.save_weights("models/monte_carlo.weights.h5")
        break

# Close the environment
env.close()
