import os
import urllib
import numpy as np
import tensorflow as tf

iris_training = 'iris_data/iris_training.csv'
iris_training_url = 'http://download.tensorflow.org/data/iris_training.csv'
iris_test = 'iris_data/iris_test.csv'
iris_test_url = 'http://download.tensorflow.org/data/iris_test.csv'

def train():
	if not os.path.exists(iris_training):
		raw = urllib.urlopen(iris_training_url).read()
		with open(iris_training,'w') as f:
			f.write(raw)

	if not os.path.exists(iris_test):
		raw = urllib.urlopen(iris_test_url).read()
		with open(iris_test, 'w') as f:
			f.write(raw)

	training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=iris_training, target_dtype=np.int, features_dtype=np.float32)
	test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=iris_test, target_dtype=np.int, features_dtype=np.float32)

	feature_columns = [tf.feature_column.numeric_column('x',shape=[4])]

	classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3, model_dir="/tmp/iris_model")

	train_input = tf.estimator.inputs.numpy_input_fn(x={'x':np.array(training_set.data)},y=np.array(training_set.target), num_epochs=None, shuffle=True)

	classifier.fit(input_fn=train_input, steps=2000)

	test_input = tf.estimator.inputs.numpy_input_fn(x={'x':np.array(test_set.data)},y=np.array(test_set.target), num_epochs=None, shuffle=False)

	# Evaluate accuracy
	accuracy_score = classifier.evaluate(input_fn=test_input)["accuracy"]
	print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

	# Classify two new flower samples
	new_samples = np.array([[6.4, 3.2, 4.5, 1.5],[5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
	predict_input = tf.estimator.inputs.numpy_input_fn(x={"x": new_samples},num_epochs=1,shuffle=False)

	predictions = list(classifier.predict(input_fn=predict_input))
	predicted_classes = [p["classes"] for p in predictions]
	print("New Samples, Class Predictions:    {}\n".format(predicted_classes))

if __name__ == '__main__':
    train()