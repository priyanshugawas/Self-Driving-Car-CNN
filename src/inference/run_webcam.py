import tensorflow.compat.v1 as tf #type: ignore
tf.disable_v2_behavior()

import cv2 #type: ignore
import os
from subprocess import call

from src.model import nvidia_model as model

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread("data/steering_wheel_image.jpg", 0)
rows, cols = img.shape

smoothed_angle = 0
cap = cv2.VideoCapture(0)

while cv2.waitKey(10) != ord('q'):
    ret, frame = cap.read()
    if not ret:
        break

    img_in = cv2.resize(frame, (200, 66)) / 255.0
    angle_rad = sess.run(model.y_pred, feed_dict={model.x: [img_in], model.keep_prob: 1.0})[0][0]
    angle_deg = angle_rad * 180 / 3.14159265

    print(f"Predicted Angle: {angle_deg:.2f} degrees")
    cv2.imshow("frame", frame)

    # smooth angle
    delta = angle_deg - smoothed_angle
    smoothed_angle += 0.2 * pow(abs(delta), 2 / 3) * (delta / abs(delta))

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("steering wheel", rotated)

cap.release()
cv2.destroyAllWindows()
