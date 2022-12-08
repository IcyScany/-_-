import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import joblib
path = u"第二次作业数据集/3_train.csv"
df = pd.read_csv(path, index_col=0)
m, n = df.shape
X = df.iloc[:, 0 : n - 1]
Y = df.iloc[:, n - 1]
X_type = list(X.columns)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
def DataVisualize(df):
    plt.figure()
    plt.subplot(311)
    plt.scatter(df.TV, df.sales,  c='r',marker='o',s=10)
    plt.grid(linestyle='-.')
    plt.subplot(312)
    plt.scatter(df.radio, df.sales,  c='b', marker='x',s=10)
    plt.grid(linestyle='-.')
    plt.ylabel("Sales")
    plt.subplot(313)
    plt.scatter(df.newspaper, df.sales,  c='y', marker='d',s=10)
    plt.xlabel('Cost')
    plt.grid(linestyle='-.')
    plt.show()

DataVisualize(df)
# 输出相关系数矩阵（最后一行为三种参数（TV，radio，newspaper）对sales的相关系数（0~1之间），越小则说明越不相关
df.corr()
# Now is (1, X1, X2, X3, X1 ^ 2, X1 * X2, X1 * X3, X2 ^ 2, X2 * X3, X3 ^ 2) 10 Features in total, so do coefs
# reg = lm.LinearRegression(copy_X=True)
reg0 = make_pipeline(PolynomialFeatures(degree=2), lm.LinearRegression(copy_X=True))
reg0.fit(X_train, Y_train)
coef = reg0[1].coef_
intercept = reg0[1].intercept_
train_score = reg0.score(X_train, Y_train)
test_score = reg0.score(X_test, Y_test)
train_mse = mean_squared_error(Y_train, reg0.predict(X_train))
test_mse = mean_squared_error(Y_test, reg0.predict(X_test))
print("Coef :", coef, "\tIntercept :", intercept)
print("Train R2 :", train_score, "\tTrain MSE :", train_mse)
print("Test R2 :", test_score, "\tTest MSE :", test_mse)
# Cross Validation Method 1
cv_results = cross_validate(reg0, X, Y, cv=KFold(n_splits=10, shuffle=True, random_state=1), scoring=['neg_mean_squared_error','r2'], return_train_score=True, return_estimator=True)
cv_results = pd.DataFrame(cv_results)
train_mse_mean = cv_results['train_neg_mean_squared_error'].mean() * (-1.0)
test_mse_mean = cv_results['test_neg_mean_squared_error'].mean() * (-1.0)
train_score_mean = cv_results['train_r2'].mean()
test_score_mean = cv_results['test_r2'].mean()
cv_results.rename(columns={'train_neg_mean_squared_error' : 'train_negMSE', 'test_neg_mean_squared_error': 'test_negMSE'}, inplace=True)
print(cv_results.drop(columns=['estimator']),"\n")
print("Mean Train MSE: %.6f, Mean Test MSE: %.6f" % (train_mse_mean, test_mse_mean))
print("Mean Train R2: %.6f, Mean Test R2: %.6f\n" % (train_score_mean, test_score_mean))

estimators = cv_results['estimator']
coef_list = []
intercept_list = []
for i in estimators:
    coef_list.append(i[1].coef_)
    intercept_list.append(i[1].intercept_)
coef_mean = np.array(coef_list).mean(axis=0)
intercept_mean = np.array(intercept_list).mean()
print("Coef :", coef_mean, "\tIntercept :", intercept_mean, "\n")
# Cross Validation Method 2
kf = KFold(n_splits=10, shuffle=True, random_state=1)
i = 1
coef_list = []
intercept_list = []
test_score_list = []
test_mse_list = []
train_score_list = []
train_mse_list = []
all_score_list = []
all_mse_list = []
for train_index, test_index in kf.split(X, Y):
    print("%d out of KFold %d" % (i, kf.n_splits), end='\t')
    print("TEST_INDEX :", test_index)
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    # reg = lm.LinearRegression(copy_X=True)
    reg = make_pipeline(PolynomialFeatures(degree=2), lm.LinearRegression(copy_X=True))
    reg.fit(X_train, Y_train)

    test_score = reg.score(X_test, Y_test)
    train_score = reg.score(X_train, Y_train)
    all_score = reg.score(X, Y)
    test_mse = mean_squared_error(Y_test, reg.predict(X_test))
    train_mse = mean_squared_error(Y_train, reg.predict(X_train))
    all_mse = mean_squared_error(Y, reg.predict(X))
    print("Train R2: %.6f, Test R2: %.6f, All R2: %.6f, Train MSE: %.6f, Test MSE: %.6f,  All MSE: %.6f" % (
    train_score, test_score, all_score, train_mse, test_mse, all_mse))

    coef_list.append(reg[1].coef_)
    intercept_list.append(reg[1].intercept_)

    test_score_list.append(test_score)
    test_mse_list.append(test_mse)
    train_score_list.append(train_score)
    train_mse_list.append(train_mse)
    all_score_list.append(all_score)
    all_mse_list.append(all_mse)

    i += 1

coef_mean = np.array(coef_list).mean(axis=0)
intercept_mean = np.array(intercept_list).mean()

test_score_mean = np.array(test_score_list).mean()
test_mse_mean = np.array(test_mse_list).mean()
train_score_mean = np.array(train_score_list).mean()
train_mse_mean = np.array(train_mse_list).mean()
all_score_mean = np.array(all_score_list).mean()
all_mse_mean = np.array(all_mse_list).mean()
# Parameters
print("\nCoef Mean: ", coef_mean)
print("Intercept Mean: ", intercept_mean, "\n")
# Mean Scores
print("Train R2 Mean: ", train_score_mean)
print("Test R2 Mean: ", test_score_mean)
print("All R2 Mean: ", all_score_mean, "\n")
print("Train MSE Mean: ", train_mse_mean)
print("Test MSE Mean: ", test_mse_mean)
print("All MSE Mean: ", all_mse_mean)
bar_width = 0.2
N = 10
plt.subplot(1, 2, 1)
plt.grid(linestyle='-.')
plt.bar(np.arange(N), height=train_score_list, width=bar_width, label='Train R2 Mean=%.6f' % train_score_mean)
plt.bar(np.arange(N) + bar_width, height=test_score_list, width=bar_width, align='center', label='Test R2 Mean=%.6f' % test_score_mean)
plt.bar(np.arange(N) + 2 * bar_width, height=all_score_list, width=bar_width, align='center', label='All R2 Mean=%.6f' % all_score_mean)
plt.xlabel('groups')
xt=range(0,10,1)
lb=['1','2','3','4','5','6','7','8','9','10']
plt.xticks(xt,lb)
plt.ylabel('$R^2$',rotation=0,fontsize=12)
plt.legend(loc='lower center')

plt.subplot(1, 2, 2)
plt.grid()
plt.bar(np.arange(N), height=train_mse_list, width=bar_width, label='Train MSE Mean=%.6f' % train_mse_mean)
plt.bar(np.arange(N) + bar_width, height=test_mse_list, width=bar_width, align='center', label='Test MSE Mean=%.6f' % test_mse_mean)
plt.bar(np.arange(N) + 2 * bar_width, height=all_mse_list, width=bar_width, align='center', label='All MSE Mean=%.6f' % all_mse_mean)
plt.xlabel('groups')
xt=range(0,10,1)
lb=['1','2','3','4','5','6','7','8','9','10']
plt.xticks(xt,lb)
plt.ylabel('$MSE$',rotation=0,fontsize=12)
plt.legend(loc='lower center')

plt.show()


def TenTimesTenFoldsCV(X, Y, polynomial=False):
    import sklearn.linear_model as lm
    from sklearn.model_selection import cross_validate, KFold
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    import numpy as np

    cv_results_dict = {}
    coef_dict = {}
    intercept_dict = {}
    estimator_dict = {}
    scoring_dict = {}

    if polynomial == True:
        reg = make_pipeline(PolynomialFeatures(degree=2), lm.LinearRegression(copy_X=True))
    else:
        reg = lm.LinearRegression()

    for i in range(10):
        # Cross Validation Method 1
        print("Times %d:" % (i + 1))
        cv_results = cross_validate(reg, X, Y, cv=KFold(n_splits=10, shuffle=True, random_state=i),
                                    scoring=['neg_mean_squared_error', 'r2'], return_train_score=True,
                                    return_estimator=True)
        cv_results = pd.DataFrame(cv_results)
        cv_results_dict[i] = cv_results

        train_mse_mean = cv_results['train_neg_mean_squared_error'].mean() * (-1.0)
        test_mse_mean = cv_results['test_neg_mean_squared_error'].mean() * (-1.0)
        train_score_mean = cv_results['train_r2'].mean()
        test_score_mean = cv_results['test_r2'].mean()
        scoring_dict[i] = {'train_mse_mean': train_mse_mean, 'test_mse_mean': test_mse_mean,
                           'train_score_mean': train_score_mean, 'test_score_mean': test_score_mean}

        cv_results.rename(
            columns={'train_neg_mean_squared_error': 'train_negMSE', 'test_neg_mean_squared_error': 'test_negMSE'},
            inplace=True)
        # If print these, results will be too long to read :(
        # print(cv_results.drop(columns=['estimator']),"\n")

        print("Mean Train MSE: %.6f, Mean Test MSE: %.6f" % (train_mse_mean, test_mse_mean))
        print("Mean Train R2: %.6f, Mean Test R2: %.6f\n" % (train_score_mean, test_score_mean))

        estimators = cv_results['estimator']
        estimator_dict[i] = estimators

        coef_list = []
        intercept_list = []
        if polynomial == True:
            for e in estimators:
                coef_list.append(e[1].coef_)
                intercept_list.append(e[1].intercept_)
        else:
            for e in estimators:
                coef_list.append(e.coef_)
                intercept_list.append(e.intercept_)

        coef_mean = np.array(coef_list).mean(axis=0)
        intercept_mean = np.array(intercept_list).mean()

        coef_dict[i] = coef_mean
        intercept_dict[i] = intercept_mean

        print("Coef :", coef_mean, "\tIntercept :", intercept_mean, "\n")
    return (cv_results_dict, coef_dict, intercept_dict, estimator_dict, scoring_dict)


_, coef_dict, intercept_dict, _, scoring_dict = TenTimesTenFoldsCV(X, Y, polynomial=True)

# coef_mean = np.array(pd.DataFrame(coef_dict).mean(axis=1))
coef_mean = np.array(list(coef_dict.values())).mean(axis=0)
intercept_mean = np.array(list(intercept_dict.values())).mean()

print("Final Coef Mean :", coef_mean)
print("Final Intercept Mean :", intercept_mean)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
reg0.coef_ = coef_mean
reg0.intercept_ = intercept_mean
# reg0.singular_ = singular_mean
Y_pred = reg0.predict(X)
print("After changing weights and bias to meaned edition:")
print("Train R2 :", reg0.score(X_train, Y_train))
print("Test R2 :", reg0.score(X_test, Y_test))
print("All R2 :", reg0.score(X, Y))
print("Train MSE :", mean_squared_error(Y_train, reg0.predict(X_train)))
print("Test MSE :", mean_squared_error(Y_test, reg0.predict(X_test)))
print("All MSE :", mean_squared_error(Y, Y_pred))
print("The rank of the variation:",reg0[1].rank_)
polynomial_items = [i.replace('X1', X_type[0]).replace('X2', X_type[1]).replace('X3', X_type[2]) for i in ['1', 'X1', 'X2', 'X3', 'X1 ^ 2', 'X1 * X2', 'X1 * X3', 'X2 ^ 2', 'X2 * X3', 'X3 ^ 2']]
finalEqStr = ""
for i in range(1, len(polynomial_items)):
	finalEqStr += "(%.6f) * %s + " % (coef_mean[i], polynomial_items[i])
finalEqStr += str(intercept_mean)
print("Final Equation : Sales =", finalEqStr, "\n")
joblib.dump(reg0, "LinearRegressor.pkl")
reg0 = joblib.load("LinearRegressor.pkl")


def tester(reg, data):
    """
    Input:
    reg: Regressor from sklearn with predict() method, e.g. sklearn.linear_model._base.LinearRegression() or Pipeline()
    data: np.array of (n, 4), e.g. array([[X1, X2, X3, Y]])
          or list e.g. [(X1, X2, X3, Y1), (X4, X5, X6, Y2)]

    Return: (Y_pred, mse)
    Y_pred: Predicted Values of Y
    mse: Mean Squared Error Value of (Y, Y_pred)
    """
    from sklearn.metrics import mean_squared_error
    data = np.array(data)  # Ensure that input data is an np.ndarray
    m, n = data.shape
    X, Y = data[:, 0: n - 1], data[:, n - 1]
    Y_pred = reg.predict(X)
    mse = mean_squared_error(Y, Y_pred)
    return (Y_pred, mse)
# Get Data Ready For Function tester()
X1 = np.array(X_test)
Y1 = np.array(Y_test)
print(X1.shape, Y1.shape)
# Y1 should be 2-dim array of (n, 1), change it now
Y1 = Y1.reshape(Y1.shape[0], 1)
data = np.hstack([X1, Y1])
print(data.shape)
#Try tester()
Y1_pred, mse1 = tester(reg0, data)
print(Y1_pred, mse1)
Y2 = Y1_pred - Y1.reshape(-1)
loss = np.sum(Y2**2) / Y2.shape[0]
print(loss)
print(mse1==loss)

fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
ax1.scatter3D(X.iloc[:, 0], X.iloc[:, 1], Y, cmap='Blues', alpha=0.6,label='origin')
ax1.plot3D(X.iloc[:, 0], X.iloc[:, 1], Y_pred, '.g', alpha=1,label='prediction')
plt.xlabel('TV')
plt.ylabel('Radio')
ax1.set_zlabel('sales')
plt.legend(loc='best')
ax1.view_init(5, -40)
plt.show()

fig = plt.figure()
ax2 = fig.add_subplot(131)
ax2.plot(X.iloc[:, 0], Y, '.r', X.iloc[:, 0], Y_pred, '.b')
plt.xlabel('TV',fontsize=12)
plt.ylabel('sales',fontsize=15)
ax2 = fig.add_subplot(132)
ax2.plot(X.iloc[:, 1], Y, '.r', X.iloc[:, 1], Y_pred, '.b')
plt.xlabel('Radio',fontsize=12)
ax2 = fig.add_subplot(133)
ax2.plot(X.iloc[:, 2], Y, '.r', X.iloc[:, 2], Y_pred, '.b')
plt.xlabel('Newspaper',fontsize=12)
plt.show()