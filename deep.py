import tensorflow as tf
import numpy as np

def input(dataset):
	return dataset.images, dataset.labels.astype(np.int32)

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

hidden_units = [256, 32, 32 ]
#hidden_units.extend([32]*300)

classifier = tf.estimator.DNNClassifier(
 feature_columns=feature_columns,
 hidden_units=hidden_units,
 optimizer=tf.train.AdamOptimizer(1e-4),
 n_classes=10,
 dropout=0.1,
 model_dir="./tmp/mnist_model"
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
 x={"x": input(mnist.train)[0]},
 y=input(mnist.train)[1],
 num_epochs=None,
 batch_size=50,
 shuffle=True
)


classifier.train(input_fn=train_input_fn, steps=10000)



# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.test)[0]},
    y=input(mnist.test)[1],
    num_epochs=1,
    shuffle=False
)

# Evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
