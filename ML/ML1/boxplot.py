import seaborn as sns
from matplotlib.ticker import MultipleLocator
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVR
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import Colours
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
import os

data = pd.read_excel("data.xlsx")
X = data[["反应温度(reaction temperature)", "离心转速(Centrifugal speed)", "超声频率(Ultrasonic frequency)", "超声时间(Ultrasonic time)"]]
Y = data[["荧光强度(Fluorescence intensity ) 106"]]





def rfc_cv(n_estimators, min_samples_split, max_features, max_depth, loss_n, data, targets):
    scaler_zscore = preprocessing.StandardScaler()
    r21 = []
    loss_n = int(round(loss_n))
    loss_ = ['mae', 'mse']
    t_size = 0.2
    for r_number in range(10):
        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size, random_state=r_number)
        sc_ale = scaler_zscore.fit(X_train)
        X_train = scaler_zscore.transform(X_train)
        X_test = scaler_zscore.transform(X_test)
        sr = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            max_features=max_features,
            max_depth=max_depth,
            criterion=loss_[loss_n],
            random_state=r_number
        )
        sr.fit(X_train, y_train)
        y_pred_test = sr.predict(X_test)
        r21.append(r2_score(y_test, y_pred_test))
    r21 = np.array(r21)
    return r21.mean()


def optimize_rf(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""

    def rfc_crossval(n_estimators, min_samples_split, max_features, max_depth, loss_n):
        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=int(round(max_features)),
            max_depth=int(max_depth),
            loss_n=loss_n,
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (10, 5000),
            "min_samples_split": (2, 25),
            "max_features": (1, 4),
            "max_depth": (2, 10),
            'loss_n': (0, 1)
        },
        random_state=1234,
        verbose=2
    )
    logger = JSONLogger(path="./randomlogs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=50, n_iter=300)
    with open('./randomlogs.json', 'a', encoding='utf-8', errors='replace')as f:
        f.write(str(optimizer.max))


def plot_r2_rf(filename, data, targets):
    with open(filename, 'r', encoding='utf-8', errors='replace')as f:
        lines = f.readlines()
        line = eval(lines[-1])
        dic = line["params"]
        # rf中的超参数设置
        n_estimators = int(dic['n_estimators'])
        min_samples_split = int(dic['min_samples_split'])
        max_features = int(round(dic['max_features']))
        max_depth = int(dic['max_depth'])
        loss_n = int(round(dic['loss_n']))
        loss_ = ['mae', 'mse']
        t_size = 0.2
        rf_train = []
        rf_train_pred = []
        rf_test = []
        rf_test_pred = []
        rf_train_aep = []
        rf_test_aep = []

        for r_number in range(1):
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size, random_state=r_number)
            scaler_zscore = preprocessing.StandardScaler()
            # 数据标准化 z-score
            sc_ale = scaler_zscore.fit(X_train)
            X_train = scaler_zscore.transform(X_train)
            X_test = scaler_zscore.transform(X_test)
            # 超参数
            sr = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_split=min_samples_split,
                max_features=max_features,
                max_depth=max_depth,
                criterion=loss_[loss_n],
                random_state=r_number
            )
            sr.fit(X_train, y_train.values.ravel())
            y_pred_test = sr.predict(X_test)
            y_pred_train = sr.predict(X_train)

            # 这里要调节 数据的形状，不然后面画图会出错

            y_train = np.array(y_train).reshape(-1, )
            y_test = np.array(y_test).reshape(-1, )

            for i in y_train:  # y_train 训练标签
                rf_train.append(i)
            for i in y_pred_train:  # y_train的预测标签
                rf_train_pred.append(i)
            for i in y_test:  # y_test真实值
                rf_test.append(i)
            for i in y_pred_test:  # y_test预测值
                rf_test_pred.append(i)
            rf_train_aep.append((abs(y_train - y_pred_train) / y_train) * 100)  # 训练数据绝对误差百分比
            rf_test_aep.append((abs(y_test - y_pred_test) / y_test) * 100)  # 测试数据绝对误差百分比

        rf_train_aep = np.array(rf_train_aep).reshape(-1, )
        rf_test_aep = np.array(rf_test_aep).reshape(-1, )
        #print(rf_train_aep.mean(),rf_test_aep.mean())

        # #data = pd.read_excel("aep1.xlsx")  # 读取表格数据
        # ax1 = fig.add_subplot(1, 1, 1)  # 编号布局
        # ax1.scatter([-0.2] * len(rf_train_aep), rf_train_aep, color='red', marker='.')  # 描点
        # ax1.scatter([0.5] * len(rf_test_aep), rf_test_aep, color='blue', marker='.')  # 描点
        # sns.set_theme(style="whitegrid")
        # ax = sns.boxplot(x="Algorithm set", y="Absolute Percentage error(%)", hue="Split", data=data)
        # # fig.savefig('picture/boxplot.png')
        # # plt.show()


def mlp(h1, h2, h3, ac, so, data, targets):
    scaler_zscore = preprocessing.StandardScaler()
    r21 = []
    ac = int(round(ac))
    so = int(round(so))
    ac_ = ['identity', 'logistic', 'tanh', 'relu']
    so_ = ['lbfgs', 'sgd', 'adam', 'sgd']
    t_size = 0.2
    for r_number in range(10):
        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size, random_state=r_number)
        sc_ale = scaler_zscore.fit(X_train)
        X_train = scaler_zscore.transform(X_train)
        X_test = scaler_zscore.transform(X_test)
        sr = MLPRegressor(hidden_layer_sizes=(h1, h2, h3), activation=ac_[ac], solver=so_[so], random_state=r_number,
                          max_iter=100000)
        sr.fit(X_train, y_train)
        y_pred_test = sr.predict(X_test)
        r21.append(r2_score(y_test, y_pred_test))
    r21 = np.array(r21)
    return r21.mean()


def optimize_mlp(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""

    def mlp_crossval(h1, h2, h3, ac, so):
        return mlp(
            h1=int(h1),
            h2=int(h2),
            h3=int(h3),
            ac=ac,
            so=so,
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=mlp_crossval,
        pbounds={
            "h1": (1, 300),
            "h2": (1, 300),
            "h3": (1, 300),
            "ac": (0, 3),
            'so': (0, 3)
        },
        random_state=1234,
        verbose=2
    )
    logger = JSONLogger(path="./MLPreglogs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=50, n_iter=300)
    with open('./MLPreglogs.json', 'a', encoding='utf-8', errors='replace')as f:
        f.write(str(optimizer.max))


def plot_r2_mlp(filename, data, targets):
    with open(filename, 'r', encoding='utf-8', errors='replace')as f:
        lines = f.readlines()
        line = eval(lines[-1])
        dic = line["params"]
        h1 = int(dic['h1'])
        h2 = int(dic['h2'])
        h3 = int(dic['h3'])
        ac = int(dic['ac'])
        so = int(dic['so'])
        loss_ = ['mae', 'mse']
        t_size = 0.2
        aep2 = []
        aep3 = []
        # r20 = []
        # mse1 = []
        # mae1 = []
        # r21 = []
        train = []
        train_pred = []
        test = []
        test_pred = []

        for r_number in range(1):
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size,
                                                                random_state=r_number)
            scaler_zscore = preprocessing.StandardScaler()
            # 数据标准化 z-score
            sc_ale = scaler_zscore.fit(X_train)
            X_train = scaler_zscore.transform(X_train)
            X_test = scaler_zscore.transform(X_test)
            # 不同的算法使用不同的回归方法
            sr = MLPRegressor(
                # h1=h1,
                # h2=h2,
                # h3=h3,
                # activation=ac,
                # solver=so,
                # criterion=loss_[loss_n],
                hidden_layer_sizes=(h1, h2, h3),
                activation="relu",
                solver='adam',
                max_iter=100000,
                random_state=r_number
            )
            sr.fit(X_train, y_train.values.ravel())
            y_pred_test = sr.predict(X_test)
            y_pred_train = sr.predict(X_train)

            # 这里要调节 数据的形状，不然后面画图会出错

            y_train = np.array(y_train).reshape(-1, )
            y_test = np.array(y_test).reshape(-1, )

            for i in y_train:  # y_train 训练标签
                train.append(i)
            for i in y_pred_train:  # y_train的预测标签
                train_pred.append(i)
            for i in y_test:  # y_test真实值
                test.append(i)
            for i in y_pred_test:  # y_test预测值
                test_pred.append(i)
            aep2.append((abs(y_train - y_pred_train) / y_train) * 100)  # 训练数据绝对误差百分比
            aep3.append((abs(y_test - y_pred_test) / y_test) * 100)  # 测试数据绝对误差百分比

        mlp_train_aep = np.array(aep2).reshape(-1, )
        mlp_test_aep = np.array(aep3).reshape(-1, )
        #print(mlp_train_aep.mean(),mlp_test_aep.mean())
        # ax1.scatter([1] * len(mlp_train_aep), mlp_train_aep, color='red', marker='.')
        # ax1.scatter([1.5] * len(mlp_test_aep), mlp_test_aep, color='blue', marker='.')
        # # plt.show()


def svc_cv(C, epsilon, loss_n, data, targets):
    scaler_zscore = preprocessing.StandardScaler()
    r21 = []
    loss_n = int(round(loss_n))
    loss_ = ['epsilon_insensitive', 'squared_epsilon_insensitive']
    t_size = 0.3
    for r_number in range(10):
        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size, random_state=r_number)
        sc_ale = scaler_zscore.fit(X_train)
        X_train = scaler_zscore.transform(X_train)
        X_test = scaler_zscore.transform(X_test)
        sr = LinearSVR(epsilon=epsilon, C=C, loss=loss_[loss_n], max_iter=1000, random_state=r_number, tol=0.01)
        sr.fit(X_train, y_train)
        y_pred_test = sr.predict(X_test)
        r21.append(r2_score(y_test, y_pred_test))
    r21 = np.array(r21)
    return r21.mean()


def plot_r2_svc(filename, data, targets):
    with open(filename, 'r', encoding='utf-8', errors='replace')as f:
        lines = f.readlines()
        line = eval(lines[-1])
        dic = line["params"]
        # loss_n = int(round(dic['loss_n']))
        epsilon = int(dic['epsilon'])
        C = int(dic['C'])
        loss_ = ['mae', 'mse']
        t_size = 0.2
        # mse0 = []
        aep4 = []
        # r20 = []
        # mse1 = []
        aep5 = []
        # r21 = []
        train = []
        train_pred = []
        test = []
        test_pred = []

        for r_number in range(1):
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size,
                                                                random_state=r_number)
            scaler_zscore = preprocessing.StandardScaler()
            # 数据标准化 z-score
            sc_ale = scaler_zscore.fit(X_train)
            X_train = scaler_zscore.transform(X_train)
            X_test = scaler_zscore.transform(X_test)
            sr = LinearSVR(
                # loss_n=loss_n,
                epsilon=epsilon,
                C=C,
                max_iter=100000
                # ac=ac,
                # so=so,
                # criterion=loss_[loss_n],
                # random_state=r_number
            )
            sr.fit(X_train, y_train)
            y_pred_test = sr.predict(X_test)
            y_pred_train = sr.predict(X_train)

            # 这里要调节 数据的形状，不然后面画图会出错

            y_train = np.array(y_train).reshape(-1, )
            y_test = np.array(y_test).reshape(-1, )

            for i in y_train:  # y_train 训练标签
                train.append(i)
            for i in y_pred_train:  # y_train的预测标签
                train_pred.append(i)
            for i in y_test:  # y_test真实值
                test.append(i)
            for i in y_pred_test:  # y_test预测值
                test_pred.append(i)
            aep4.append((abs(y_train - y_pred_train) / y_train) * 100)  # 训练数据绝对误差百分比
            aep5.append((abs(y_test - y_pred_test) / y_test) * 100)  # 测试数据绝对误差百分比

        svc_train_aep = np.array(aep4).reshape(-1, )
        svc_test_aep = np.array(aep5).reshape(-1, )
        # print(svc_train_aep, svc_test_aep)
        # ax1.scatter([2] * len(svc_train_aep), svc_train_aep, color='red', marker='.')
        # ax1.scatter([2.5] * len(svc_test_aep), svc_test_aep, color='blue', marker='.')
        # plt.show()


def optimize_svc(data, targets):
    """Apply Bayesian Optimization to SVC parameters."""

    def svc_crossval(C, epsilon, loss_n):
        return svc_cv(C=C, epsilon=epsilon, loss_n=loss_n, data=data, targets=targets)

    optimizer = BayesianOptimization(
        f=svc_crossval,
        pbounds={"C": (0.0001, 1000), "epsilon": (0, 0.2), 'loss_n': (0, 1)},
        random_state=1234,
        verbose=2
    )
    logger = JSONLogger(path="./linsvrlogs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=50, n_iter=300)
    with open('./linsvrlogs.json', 'a', encoding='utf-8', errors='replace')as f:
        f.write(str(optimizer.max))


def plot_r2_ssvr(filename, data, targets):
    with open(filename, 'r', encoding='utf-8', errors='replace')as f:
        lines = f.readlines()
        line = eval(lines[-1])
        dic = line["params"]
        # loss_n = int(round(dic['loss_n']))
        ker = int(dic['ker'])
        epsilon = int(dic['epsilon'])
        C = int(dic['C'])
        loss_ = ['mae', 'mse']
        t_size = 0.2
        mse0 = []
        svr_train_aep = []
        r20 = []
        mse1 = []
        svr_test_aep = []
        r21 = []
        train = []
        train_pred = []
        test = []
        test_pred = []

        for r_number in range(1):
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size,
                                                                random_state=r_number)
            scaler_zscore = preprocessing.StandardScaler()
            # 数据标准化 z-score
            sc_ale = scaler_zscore.fit(X_train)
            X_train = scaler_zscore.transform(X_train)
            X_test = scaler_zscore.transform(X_test)
            sr = SVR(
                # loss_n=loss_n,
                epsilon=epsilon,
                C=C,
                # kernel=ker
                # ac=ac,
                # so=so,
                # criterion=loss_[loss_n],
                # random_state=r_number
            )
            sr.fit(X_train, y_train.values.ravel())
            y_pred_test = sr.predict(X_test)
            y_pred_train = sr.predict(X_train)

            # 这里要调节 数据的形状，不然后面画图会出错

            y_train = np.array(y_train).reshape(-1, )
            y_test = np.array(y_test).reshape(-1, )

            for i in y_train:  # y_train 训练标签
                train.append(i)
            for i in y_pred_train:  # y_train的预测标签
                train_pred.append(i)
            for i in y_test:  # y_test真实值
                test.append(i)
            for i in y_pred_test:  # y_test预测值
                test_pred.append(i)

            svr_train_aep.append((abs(y_train - y_pred_train) / y_train) * 100)  # 训练数据绝对误差百分比
            svr_test_aep.append((abs(y_test - y_pred_test) / y_test) * 100)  # 测试数据绝对误差百分比

        svr_train_aep = np.array(svr_train_aep).reshape(-1, )
        svr_test_aep = np.array(svr_test_aep).reshape(-1, )
        print(svr_train_aep.mean(), svr_test_aep.mean())
        # result_list = svr_test_aep
        # columns = ["BOA-SVR"]
        # dt = pd.DataFrame(result_list, columns=columns)
        # dt.to_excel("svr_test_aep.xlsx", index=0)


def optimize_ssvr(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""

    def ssvr_crossval(C, epsilon, ker):
        return svr(
            C=C,
            epsilon=epsilon,
            ker=ker,
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=ssvr_crossval,
        pbounds={
            "C": (0.0001, 1000),
            "epsilon": (0, 0.2),
            "ker": (0, 3),
        },
        random_state=1234,
        verbose=2
    )
    logger = JSONLogger(path="./SVRmaclogs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=50, n_iter=300)
    with open('./SVRmaclogs.json', 'a', encoding='utf-8', errors='replace')as f:
        f.write(str(optimizer.max))


def svr(C, epsilon, ker, data, targets):
    scaler_zscore = preprocessing.StandardScaler()
    r21 = []
    ker = int(round(ker))
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    t_size = 0.2
    for r_number in range(5):
        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size, random_state=r_number)
        sc_ale = scaler_zscore.fit(X_train)
        X_train = scaler_zscore.transform(X_train)
        X_test = scaler_zscore.transform(X_test)
        sr = SVR(epsilon=epsilon, C=C, kernel=kernel[ker], max_iter=100000)
        sr.fit(X_train, y_train)
        y_pred_test = sr.predict(X_test)
        r21.append(r2_score(y_test, y_pred_test))
    r21 = np.array(r21)
    return r21.mean()


if __name__ == '__main__':
    #optimize_rf(X, targets=Y)
    # optimize_mlp(X, targets=Y)
    # optimize_svc(X, targets=Y)
     #optimize_ssvr(X, targets=Y)
    #plot_r2_rf("randomlogs.json", X, Y)
     #plot_r2_mlp("MLPreglogs.json", X, Y)
    # plot_r2_svc("linsvrlogs.json", X, Y)
    plot_r2_ssvr("SVRmaclogs.json", X, Y)
    # plot_optimization("randomlogs.json")
    # plot_optimization("MLPreglogs.json")
    # plot_optimization("linsvrlogs.json")
    # plot_optimization("SVRmaclogs.json")
