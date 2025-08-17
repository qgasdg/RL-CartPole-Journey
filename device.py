import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print(f"텐서플로우가 다음 GPU를 사용 중입니다: {gpus}")
else:
    print("텐서플로우가 GPU를 찾지 못했습니다. CPU로 실행 중입니다.")
