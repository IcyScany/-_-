[ReadMe]

The TestInterface.py contains 4 test functions:

1. ClassifierTest(modelpath, filepath)
	modelpath: Model File's Path (.pkl)
	filepath: Test Data File's Path (Caution: Format should be the same as X_train_classification.csv)
		Returns: Accuracy

2. RegressorTest(modelpath, filepath)
	modelpath: Model File's Path (.pkl)
	filepath: Test Data File's Path (Caution: Format should be the same as X_train_regression.csv)
		Returns: Mean Absolute Percentage Error

3. ClassifierTestXY(modelpath, X_test, Y_test)
	modelpath: Model File's Path (.pkl)
	X_test: Test Data of X Columns, shape: (m, n) of np.ndarray / pd.DataFrame
	Y_test: Test Data of Y Columns, shape: (m,) of np.ndarray / pd.Series
		Returns: Accuracy

4. RegressorTestXY(modelpath, X_test, Y_test)
	modelpath: Model File's Path (.pkl)
	X_test: Test Data of X Columns, shape: (m, n) of np.ndarray / pd.DataFrame
	Y_test: Test Data of Y Columns, shape: (m,) of np.ndarray / pd.Series
		Returns: Mean Absolute Percentage Error
		
Or you can just run "acc_test.py" to get Classifier's Accuracy, "mape_test.py" to get Regressor's Score.

[Caution]

1. Ensure that you have set the modelpath in "mape_test.py" before running it! 