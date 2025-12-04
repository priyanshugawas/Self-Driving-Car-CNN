import cv2 #type:ignore
import random

# Lists for image paths and labels
xs = []
ys = []

# Batch pointers
train_batch_pointer = 0
val_batch_pointer = 0

# Load metadata
with open("data/driving_dataset/data.txt") as f:
    for line in f:
        img_path, angle_deg = line.split()
        xs.append("data/driving_dataset/" + img_path)
        ys.append(float(angle_deg) * 3.14159265 / 180.0)  # deg → radians

# Shuffle dataset
combined = list(zip(xs, ys))
random.shuffle(combined)
xs, ys = zip(*combined)

# Train-validation split
split_idx = int(0.8 * len(xs))
train_xs, train_ys = xs[:split_idx], ys[:split_idx]
val_xs, val_ys = xs[split_idx:], ys[split_idx:]

num_train = len(train_xs)
num_val = len(val_xs)


def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out, y_out = [], []

    for i in range(batch_size):
        idx = (train_batch_pointer + i) % num_train
        img = cv2.imread(train_xs[idx])[-150:]
        img = cv2.resize(img, (200, 66)) / 255.0
        x_out.append(img)
        y_out.append([train_ys[idx]])

    train_batch_pointer += batch_size
    return x_out, y_out


def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out, y_out = [], []

    for i in range(batch_size):
        idx = (val_batch_pointer + i) % num_val
        img = cv2.imread(val_xs[idx])[-150:]
        img = cv2.resize(img, (200, 66)) / 255.0
        x_out.append(img)
        y_out.append([val_ys[idx]])

    val_batch_pointer += batch_size
    return x_out, y_out
