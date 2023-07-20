import numpy as np
import glob
import os 
import cv2

from keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

store_image = []
train_path = './train'
fps = 10  
train_videos = os.listdir(train_path)
train_images_path = train_path + '/frames'
os.makedirs(train_images_path, exist_ok=True)

def store_inarray(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_AREA)
    gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    store_image.append(gray)

for video in train_videos:
    os.system('ffmpeg -i {}/{} -vf fps={}/1 {}/frames/%03d.jpg'.format(train_path, video, fps, train_path))  # Adjust the ffmpeg command to capture more frames
images = os.listdir(train_images_path)
for image in images:
    image_path = os.path.join(train_images_path, image)
    store_inarray(image_path)

store_image = np.array(store_image)
a, b, c = store_image.shape
store_image.resize(b, c, a)
store_image = (store_image - store_image.mean()) / (store_image.std())
store_image = np.clip(store_image, 0, 1)
np.save('training.npy', store_image)

stae_model = Sequential()

num_frames = 10

stae_model.add(
    Conv3D(filters=128, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid', input_shape=(227, 227, num_frames, 1),
           activation='tanh'))
stae_model.add(Conv3D(filters=64, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tanh'))
stae_model.add(
    ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', dropout=0.4, recurrent_dropout=0.3,
               return_sequences=True))
stae_model.add(
    ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', dropout=0.3, return_sequences=True))
stae_model.add(
    ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, return_sequences=True, padding='same', dropout=0.5))
stae_model.add(
    Conv3DTranspose(filters=128, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tanh'))
stae_model.add(
    Conv3DTranspose(filters=1, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid', activation='tanh'))

stae_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

training_data = np.load('training.npy')
frames = training_data.shape[2]
frames = frames - frames % num_frames

training_data = training_data[:, :, :frames]
training_data = training_data.reshape(-1, 227, 227, num_frames)
training_data = np.expand_dims(training_data, axis=4)
target_data = training_data.copy()

epochs = 300
batch_size = 1

callback_save = ModelCheckpoint("saved_model.h5", monitor="mean_squared_error", save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

stae_model.fit(training_data, target_data, batch_size=batch_size, epochs=epochs,
               callbacks=[callback_save, callback_early_stopping], initial_epoch=0)

stae_model.save("saved_model.h5")

validation_data = np.load('training.npy')  

evaluation = stae_model.evaluate(training_data, target_data)
print('Loss:', evaluation[0])
print('Accuracy:', evaluation[1])
