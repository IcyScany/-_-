import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

mpl.rcParams["font.sans-serif"] = [u"SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

path = u"第一次作业数据集/3_train.csv"
df = pd.read_csv(path,header=None)
X = df[[0,1]]
Y = df[[2]]
X = np.array(X)
Y = np.array(Y)
mean_X1 = X[:,0].mean()
mean_X2 = X[:,1].mean()
std_X1 = X[:,0].std()
std_X2 = X[:,1].std()
X[:,0] = (X[:,0] - mean_X1) / std_X1
X[:,1] = (X[:,1] - mean_X2) / std_X2
X = np.hstack([X,np.ones((X.shape[0], 1))])
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
loss_list = []
test_loss_list = []
epochs_list = []

def sigmoid(x):
	"""
	return sigmid(x)
	"""
	return 1.0 / (1.0 + np.exp(-x))
	
def loss_func(xMat, yMat, weights):
	"""
	计算损失函数
	xMat: 特征数据矩阵
	weights: 参数
	yMat: 标签数据矩阵
	return: 损失函数
	"""
	m, n = xMat.shape
	hypothesis = sigmoid(np.dot(xMat, weights))  # 预测值
	loss = (-1.0 / m) * np.sum(yMat.T * np.log(hypothesis) + (1 - yMat).T * np.log(1 - hypothesis))  # 损失函数
	return loss

def SGD(data_x, data_y, test_x, test_y, alpha=0.01, max_epochs=50000,epsilion=1e-8):
	xMat = np.mat(data_x)
	yMat = np.mat(data_y)
	xMat_test = np.mat(test_x)
	yMat_test = np.mat(test_y)
	m, n = xMat.shape
	weights = np.ones((n, 1))  # 模型参数
	epochs_count = 0
	global loss_list
	global test_loss_list
	global epochs_list
	while epochs_count < max_epochs:
		rand_i = np.random.randint(m)  # 随机取一个样本
		
		loss = loss_func(xMat, yMat, weights) #前一次迭代的损失值
		test_loss = loss_func(xMat_test, yMat_test, weights)
		hypothesis = sigmoid(np.dot(xMat[rand_i,:], weights)) #预测值
		error = hypothesis - yMat[rand_i,:] #预测值与实际值误差
		
		grad = np.dot(xMat[rand_i,:].T, error) #损失函数的梯度
		weights = weights - alpha * grad #参数更新
		loss_new = loss_func(xMat, yMat, weights)#当前迭代的损失值
		test_loss_new = loss_func(xMat_test, yMat_test, weights)
		print(loss_new)
		
		if abs(loss_new-loss)<epsilion:
			break
		loss_list.append(loss_new)
		test_loss_list.append(test_loss_new)
		epochs_list.append(epochs_count)
		epochs_count += 1
		
	print(f'Final Train Loss = {loss_new}')
	print(f'Final Test Loss = {test_loss_new}')
	print('迭代到第{}次，结束迭代！'.format(epochs_count))
	return weights

def Acc(weights, test_x, test_y):
	xMat_test = np.mat(test_x)
	m, n = xMat_test.shape
	result = []
	for i in range(m):
		proba = sigmoid(np.dot(xMat_test[i,:], weights)) #预测值
		if proba < 0.5:
			predict = 0
		else:
			predict = 1
		result.append(predict)
	acc = (np.array(result)==test_y.squeeze()).mean()
	return acc

def GetCurveParams(data_y, y_pred, theta=0.001):
	k = 1 / theta
	m, _ = data_y.shape
	x_ = []
	y_ = []
	z_ = []
	point5_info = {}
	auc = 0.0
	prev_z = 0
	for i in range(1, int(k) + 1):
		tp = 0
		fp = 0
		tn = 0
		fn = 0
		p = theta * i
		for j in range(m):
			if y_pred[0, j] >= p and data_y[j, 0] == 1:
				tp += 1
			elif y_pred[0, j] >= p and data_y[j, 0] == 0:
				fp += 1
			elif y_pred[0, j] < p and data_y[j, 0] == 1:
				fn += 1
			elif y_pred[0, j] < p and data_y[j, 0] == 0:
				tn += 1
			else:
				pass
		
		# for label 1
		recall = tp / (tp + fn)
		precision = (tp / (tp + fp)) if (tp + fp) != 0 else 1
		fpr = fp / (fp + tn)
		f1_score = 2 * precision * recall / (precision + recall)
		# for label 0
		recall0 = tn  / (tn + fp)
		precision0 = (tn / (tn + fn)) if (tn + fn) != 0 else 1
		f1_score0 = 2 * precision0 * recall0 / (precision0 + recall0)
		x_.append(recall)# Recall & TPR [Ascending]
		y_.append(precision) # Precision
		z_.append(fpr) # FPR [Descending]
		
		if p == 0.50:
			accuracy = (tp + tn) / (tp + tn + fp + fn)
			point5_info={'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'precision': precision, 'recall': recall, 
			'f1-score': f1_score, 'pos': tp + fn, 'precision0': precision0, 'recall0': recall0, 
			'f1-score0': f1_score0, 'neg': tn + fp, 'accuracy': accuracy}
		
		if z_[-1] != prev_z:
			auc += (z_[-1] - prev_z) * x_[-1] # Inverse Sum-ing
			prev_z = z_[-1]
	point5_info['auc'] = 1 - auc
	return (x_[::-1], y_[::-1], z_[::-1], point5_info)

def Visualizer(data_x, data_y, weights):
	
	ax = plt.subplot(2, 2, 1)
	ax.plot(data_x[data_y.reshape(-1)==0,0], data_x[data_y.reshape(-1)==0,1], 'or', label='1')
	ax.plot(data_x[data_y.reshape(-1)==1,0], data_x[data_y.reshape(-1)==1,1], 'ob', label='0')
	x = np.arange(-2.5, 2.5, 0.01)
	y = (-weights[2] - weights[0] * x) / weights[1]
	ax.plot(x, y.T)
	plt.legend(loc='best')
	plt.title(f'Function: {weights[0]} * X1 + {weights[1]} * X2 + {weights[2]}')
	plt.xlabel('X1')
	plt.ylabel('X2')
	
	ax = plt.subplot(2, 2, 2)
	plt.plot(epochs_list, loss_list, label='Train Loss')
	plt.plot(epochs_list, test_loss_list, label='Test loss')
	plt.legend(loc='best')
	plt.title('Loss Function')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	
	
	y_pred = weights[0] * data_x[:,0] + weights[1] * data_x[:,1] + weights[2]
	y_pred_log = np.array(sigmoid(y_pred))
	Recall, Precision, fpr, point5_info = GetCurveParams(data_y, y_pred_log)
	tpr = Recall.copy()
	thr = np.sort(y_pred_log)[::-1].tolist()
	Recall.append(1)
	Precision.append(0)
	
	ax = plt.subplot(2, 2, 3)
	plt.plot(Recall, Precision, 'k')
	plt.title('PR Curve')
	# plt.plot([(0, 0), (1, 1)], 'r--')
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.ylabel('Precision')
	plt.xlabel('Recall')
	
	ax = plt.subplot(2, 2, 4)
	plt.plot(fpr, tpr, 'k', label='AUC = %.6f'%point5_info['auc'])
	plt.title('Receiver Operating Characteristic')
	plt.plot([(0, 0),(1, 1)],'r--')
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.legend(loc='best')
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	
	df = pd.DataFrame(np.array([[point5_info['precision'], point5_info['recall'], point5_info['f1-score'], point5_info['pos']],
							   [point5_info['precision0'], point5_info['recall0'], point5_info['f1-score0'], point5_info['neg']]]),
							   columns=['precision', 'recall', 'f1-score', 'support'],
							   index=['1.0', '0.0'])
	print(df)
	print('AUC: ', point5_info['auc'])
	print('Accuracy: ', point5_info['accuracy'])
	print('Loss: ', loss_func(data_x, data_y, weights))
	
	plt.show()
	
def main():
	weights = SGD(X_train, Y_train, X_test, Y_test)
	acc = Acc(weights, X_test, Y_test)
	print("Test accuracy: ", acc)
	Visualizer(X, Y, weights)
	
if __name__=='__main__':
	main()