import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
import numpy as np
import os

MODEL_SAVE_FOLDER_PATH = './model/'

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)


# 훈련 데이터 로딩 및 전처리
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# 모델 정의
model = Sequential([
    # CNN 레이어 추가
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    # Dense 레이어 추가
    Dense(256, activation='relu'),
    Dropout(0.3),  # 드롭아웃 비율을 조정하여 사용 (0.5는 일반적으로 사용되는 값)
    Dense(10, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.summary()

# 사용자 정의 전처리 함수
def add_noise(image):
    positive_values = np.random.uniform(0.1, 0.4)
    image += positive_values
    image = np.clip(image, 0, 1)
    return image

# 모델 훈련
myModel = model.fit(x_train, y_train,
                    epochs=5,
                    validation_data=(x_test, y_test))
# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)



model_path = MODEL_SAVE_FOLDER_PATH + 'mnist-' + str(test_acc) + "-myModel.h5"
# 모델 저장 (파일 경로 설정)
model.save(model_path)
