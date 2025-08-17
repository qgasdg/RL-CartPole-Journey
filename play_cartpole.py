import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

import gymnasium as gym

from PIL import Image

import os

# --- 1. 훈련 때와 똑같은 구조의 모델 생성 ---
# 이 구조가 가중치 파일이 만들어졌을 때의 구조와 정확히 일치해야 합니다.
num_states = 4
num_actions = 2

inputs = layers.Input(shape=(num_states,))
common = layers.Dense(128, activation="relu")(inputs)
action_probs_output = layers.Dense(num_actions, activation="softmax")(common)
critic_output = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action_probs_output, critic_output])

# --- 2. 저장해둔 가중치(.h5 파일) 불러오기 ---
# 파일 이름이 다르면 이 부분을 수정해주세요.
# weights_filename = "models/TD_actor_critic.weights.h5"
weights_filename = "models/cartpole_train.weights.h5"

if not os.path.exists(weights_filename):
    print(f"오류: 가중치 파일 '{weights_filename}'을 찾을 수 없습니다.")
    print("먼저 훈련 코드를 실행하여 모델을 저장해주세요.")
else:
    model.load_weights(weights_filename)
    print(f"가중치 '{weights_filename}'를 성공적으로 불러왔습니다.")

    # --- 3. 렌더링을 위한 환경 생성 ---
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # --- 4. 한 에피소드를 플레이하고 프레임을 녹화하는 함수 ---
    def render_episode(env: gym.Env, model: keras.Model, max_steps: int):
        state, _ = env.reset(seed=42)  # 항상 같은 플레이를 보기 위해 시드 고정
        images = []

        # 첫 프레임 렌더링 및 저장
        screen = env.render()
        images.append(Image.fromarray(screen))

        for i in range(1, max_steps + 1):
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)

            # 모델을 사용해 행동 확률 예측 (크리틱 값은 필요 없으므로 _ 로 무시)
            action_probs = model(state_tensor)

            # '탐험'이 필요 없는 플레이 단계에서는 가장 확률이 높은 행동을 선택
            action = np.argmax(np.squeeze(action_probs))

            # 환경에서 행동 실행
            state, reward, done, _, _ = env.step(action)

            # 화면 렌더링 및 프레임 저장
            screen = env.render()
            images.append(Image.fromarray(screen))

            if done:
                break

        return images

    # --- 5. GIF 생성 실행 ---
    print("GIF 생성을 시작합니다...")
    # CartPole-v1은 500스텝을 버티면 성공이므로 max_steps를 500으로 설정
    images = render_episode(env, model, 500)

    # 이미지 리스트를 사용해 GIF 파일 저장
    images[0].save(
        "cartpole_play.gif",
        save_all=True,
        append_images=images[1:],
        loop=0,  # 0: 무한 반복
        duration=30,  # 각 프레임의 재생 시간 (ms)
    )

    print("성공! 'cartpole_play.gif' 파일이 현재 폴더에 저장되었습니다.")
