import cv2
import numpy as np
from tensorflow.keras.models import load_model
import argparse
from PIL import Image
import imutils

def mean_squared_loss(x1, x2):
    difference = x1 - x2
    a, b, c, d, e = difference.shape
    n_samples = a * b * c * d * e
    sq_difference = difference ** 2
    Sum = sq_difference.sum()
    distance = np.sqrt(Sum)
    mean_distance = distance / n_samples

    return mean_distance

model = load_model("saved_model.h5")

cap = cv2.VideoCapture(0)  # 0 for default camera, you can also specify a camera index if multiple cameras are present
print(cap.isOpened())

while cap.isOpened():
    imagedump = []
    ret, frame = cap.read()

    frame = imutils.resize(frame, width=1000, height=1200)

    for i in range(10):
        gray = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
        gray = 0.2989 * gray[:, :, 0] + 0.5870 * gray[:, :, 1] + 0.1140 * gray[:, :, 2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        imagedump.append(gray)

    imagedump = np.array(imagedump)

    imagedump.resize(227, 227, 10)
    imagedump = np.expand_dims(imagedump, axis=0)
    imagedump = np.expand_dims(imagedump, axis=4)

    output = model.predict(imagedump)

    loss = mean_squared_loss(imagedump, output)

    if loss > 0.00068:
        print('Normal Event')
        cv2.putText(frame, "Normal Event", (220, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        frame = cv2.copyMakeBorder(frame, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(0, 255, 0))
    else:
        print('Abnormal Event Detected')
        cv2.putText(frame, "Abnormal Event", (220, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        frame = cv2.copyMakeBorder(frame, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(0, 0, 255))

    cv2.imshow("video", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
