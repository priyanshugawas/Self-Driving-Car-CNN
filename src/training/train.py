import os
import tensorflow.compat.v1 as tf #type: ignore
tf.disable_v2_behavior()

from tensorflow.core.protobuf import saver_pb2
from src.data import driving_data
from src.model import nvidia_model as model

LOGDIR = "save"

sess = tf.InteractiveSession()

# Loss with L2 regularization
L2_CONST = 0.001
train_vars = tf.trainable_variables()

loss = tf.reduce_mean(tf.square(model.y_true - model.y_pred)) + \
       L2_CONST * tf.add_n([tf.nn.l2_loss(v) for v in train_vars])

optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

# TensorBoard
tf.summary.scalar("loss", loss)
merged_summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("logs", sess.graph)

saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)

epochs = 30
batch_size = 100

for epoch in range(epochs):
    steps_per_epoch = driving_data.num_train // batch_size

    for step in range(steps_per_epoch):
        xs, ys = driving_data.LoadTrainBatch(batch_size)
        sess.run(optimizer, feed_dict={model.x: xs, model.y_true: ys, model.keep_prob: 0.8})

        if step % 10 == 0:
            xs_val, ys_val = driving_data.LoadValBatch(batch_size)
            val_loss = sess.run(loss, feed_dict={model.x: xs_val, model.y_true: ys_val, model.keep_prob: 1.0})
            print(f"Epoch {epoch}, Step {step}, Loss {val_loss}")

        summary = sess.run(merged_summary, feed_dict={model.x: xs, model.y_true: ys, model.keep_prob: 1.0})
        summary_writer.add_summary(summary, epoch * steps_per_epoch + step)

    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
    print(f"Model saved at epoch {epoch}")

print("Run TensorBoard with:\n tensorboard --logdir=logs")
