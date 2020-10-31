import csv
import cv2
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data_dir = "/opt/carnd_p3/data/"
data_path = os.path.join(data_dir, "driving_log.csv")


def get_logs(data_path=os.path.join(data_dir, "driving_log.csv")):
    logs = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the headers
        for log_line in reader:
            # Latest data is stored using the absolute path instead of the relative one
            log_line[0] = log_line[0].replace(data_dir, "")
            log_line[1] = log_line[1].replace(data_dir, "")
            log_line[2] = log_line[2].replace(data_dir, "")
            logs.append(log_line)
    return logs


def generator(samples, batch_size=32):
    num_samples = len(samples)
    n_cameras = 3
    correction_base = 0.1
    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Image centre
                # cv2 reads the image in BGR but drive.py uses RGB
                steering_center = float(batch_sample[3])
                for i in range(n_cameras):
                    #  Center i = 0, left i = 1, right i = 2
                    correction = 0 if i == 0 else correction_base + np.random.uniform(low=0, high=0.2)
                    direction = -1 if i == 2 else 1
                    steering = steering_center + correction * direction

                    img_bgr = cv2.imread(os.path.join(data_dir, batch_sample[i].strip()))
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    angles.append(steering)

                    # Flip images and angles
                    img_flipped = np.fliplr(img)
                    steering_flipped = - steering
                    images.append(img_flipped)
                    angles.append(steering_flipped)

                    # Apply histogram equalization to improve contrast
                    # https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
                    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
                    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                    steering_eq = steering
                    images.append(img_eq)
                    angles.append(steering_eq)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


EPOCHS = 3
BATCH_SIZE = 64
DROPOUT_RATE = 0.75

# compile and train the model using the generator function
logs = get_logs(data_path)
train_samples, validation_samples = train_test_split(logs, test_size=0.2, shuffle=True, random_state=31415)
print("Number of train samples:", len(train_samples))
print("Number of validation samples:", len(validation_samples))
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

# Model based on the NVIDIA "End to End Learning for Self-Driving Cars" publication
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding="VALID", activation="relu"))
model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding="VALID", activation="relu"))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding="VALID", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", activation="relu"))
model.add(Dropout(rate=DROPOUT_RATE))
model.add(Flatten())
model.add(Dense(units=100, activation="relu"))
model.add(Dense(units=50, activation="relu"))
model.add(Dense(units=10, activation="relu"))
model.add(Dense(units=1))

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(len(train_samples)/BATCH_SIZE),
    validation_data=validation_generator,
    validation_steps=np.ceil(len(validation_samples)/BATCH_SIZE),
    callbacks=[early_stop],
    epochs=EPOCHS,
    verbose=1
)

model.save('model.h5')
print("Model saved")
print('Loss')
print(history.history['loss'])
print('Validation Loss')
print(history.history['val_loss'])
