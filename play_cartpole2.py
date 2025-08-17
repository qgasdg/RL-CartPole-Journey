import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gym
from PIL import Image
import os

# --- 1. 저장된 모델 전체를 불러오기 ---
model_filename = "models/cartpole_train.weights.h5"

if not os.path.exists(model_filename):
    print(f"오류: 모델 파일 '{model_filename}'을 찾을 수 없습니다.")
    print("먼저 train_reinforce.py를 실행하여 모델을 훈련/저장해주세요.")
else:
    print(f"모델 '{model_filename}'을 성공적으로 불러왔습니다.")

    # --- [변경 후] ---
    # 1. 훈련 때와 똑같은 '텅 빈' 모델 구조를 먼저 만듭니다.
    #    (이 구조가 가중치를 저장할 때의 구조와 100% 동일해야 합니다.)
    num_states = 4
    num_actions = 2
    inputs = layers.Input(shape=(num_states,))
    common = layers.Dense(128, activation="relu")(inputs)
    action = layers.Dense(num_actions, activation="softmax")(common)
    # 만약 액터-크리틱 모델을 저장했다면 크리틱 출력도 필요합니다.
    # critic = layers.Dense(1)(common)
    # model = keras.Model(inputs=inputs, outputs=[action, critic])
    # 만약 액터만 있는 REINFORCE 모델이었다면 아래 코드를 사용하세요.
    model = keras.Model(inputs=inputs, outputs=action)

    # 2. 그 구조 안으로 가중치 데이터만 불러옵니다.
    model.load_weights(model_filename)

    # --- 2. 렌더링을 위한 환경 생성 ---
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # --- 3. 한 에피소드를 플레이하고 프레임을 녹화하는 함수 ---
    def render_episode(env: gym.Env, model: keras.Model, max_steps: int):
        state, _ = env.reset(seed=42)
        images = []

        screen = env.render()
        images.append(Image.fromarray(screen))

        for i in range(1, max_steps + 1):
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)

            # 모델을 사용해 행동 확률 예측
            action_probs = model(state_tensor)

            # 플레이 시에는 가장 확률이 높은 행동을 선택
            action = np.argmax(np.squeeze(action_probs))

            state, reward, done, _, _ = env.step(action)

            screen = env.render()
            images.append(Image.fromarray(screen))

            if done:
                break

        return images

    # --- 4. GIF 생성 실행 ---
    print("GIF 생성을 시작합니다...")
    images = render_episode(env, model, 500)

    images[0].save(
        "cartpole_reinforce_play.gif",
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=30,
    )

    print("성공! 'cartpole_reinforce_play.gif' 파일이 현재 폴더에 저장되었습니다.")
