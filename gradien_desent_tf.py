import tensorflow as tf
import numpy as np

theta = tf.Variable(-5.0)
eps=0.0001

@tf.function
def Loss(theta):
  return theta**2+2*theta+4

opt = tf.keras.optimizers.SGD(learning_rate=0.1)

while True:
  with tf.GradientTape() as tape:
    loss=Loss(theta)
    print(loss)
  grads=tape.gradient(loss,[theta])
  print(grads)
  opt.apply_gradients(zip(grads,[theta]))
  if abs(grads[0].numpy()) < eps:
    break

print(theta)