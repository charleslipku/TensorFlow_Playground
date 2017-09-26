import os
import urllib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import DNNClassifier

def main():
	training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
		filename='./iris_data/iris_training.csv',
        target_dtype=np.int,
        features_dtype=np.float32
	)
	test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
		filename='./iris_data/iris_test.csv',
        target_dtype=np.int,
        features_dtype=np.float32
	)

	feature_columns = [tf.feature_column.numeric_column("x",shape=[4])]

	clf = DNNClassifier(hidden_units=[10,20,10],feature_columns=feature_columns,model_dir='./iris_model',n_classes=3)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":np.array(training_set.data)},
	                                                    y=np.array(training_set.target),
	                                                    num_epochs=None,
	                                                    shuffle=True)

	clf.fit(input_fn=train_input_fn, steps=2000)

	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(test_set.data)},
		y=np.array(test_set.target),
		num_epochs=1,
		shuffle=False)

	accuracy_score = clf.evaluate(input_fn=test_input_fn)["accuracy"]
	print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

	new_samples = np.array([[6.4, 3.2, 4.5, 1.5],[5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
	predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": new_samples},num_epochs=1,shuffle=False)
	predictions = list(clf.predict(input_fn=predict_input_fn))
	print predictions

	print("New Samples, Class Predictions:    {}\n".format(predictions))

if __name__ == '__main__':
    main()
