import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from util import trainBP, plotBP, plot_net_predict_result
from ga.ga import GA

if __name__ == '__main__':
    csv_data = pd.read_csv("./累计成交额含价格.csv", header=None)

    # 数据预处理
    x = csv_data.iloc[:, 1:-1].values
    y = csv_data.iloc[:, -1].values
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    print(x.shape)
    print(y.shape)
    print(x.dtype)
    print(y.dtype)

    # ----------------------------BP网络超参数----------------------------
    shuffle_seed = 233
    test_set_size = 10
    hidden_number = 5  # 仅有一层隐藏层，神经元个数为5
    learning_rate = 2e-3
    epochs = 2
    BATCH_SIZE = 32

    # 获取训练集和测试集
    x_train = x[:-test_set_size]
    x_test = x[-test_set_size:]
    y_train = y[:-test_set_size]
    y_test = y[-test_set_size:]
    print(y_train.shape)
    y_train = y_train.reshape((y_train.shape[0], -1))
    y_test = y_test.reshape((y_test.shape[0], -1))
    print(y_train.shape)

    # MinMax归一化
    feature_sc = MinMaxScaler(feature_range=(-1, 1))
    x_train_scaled = feature_sc.fit_transform(x_train)
    x_test_scaled = feature_sc.transform(x_test)
    target_sc = MinMaxScaler(feature_range=(-1, 1))
    y_train_scaled = target_sc.fit_transform(y_train)
    y_test_scaled = target_sc.transform(y_test)

    loss_fn = torch.nn.MSELoss(size_average=True, reduce=True)

    net, train_loss, test_loss = trainBP(x_train_scaled.shape[1], hidden_number, 1,
                                         learning_rate, epochs, shuffle_seed, BATCH_SIZE, loss_fn,
                                         x_train_scaled, y_train_scaled,
                                         x_test_scaled, y_test_scaled)
    plotBP(train_loss, test_loss)
    plot_net_predict_result(net, x_test_scaled, target_sc, y_test)

    #遗传算法参数
    popsize = 100; #遗传算法种群数
    iter_max = 6; #遗传算法迭代次数
    PM = 0.05; #变异概率
    PC = 0.7; #交叉概率
    input_number = 21#输入维度
    output_number = 1#输出维度
    gaObject = GA(net, input_number, hidden_number, output_number, popsize, iter_max, PM, PC)
    gaObject.gafun(x_train_scaled, y_train_scaled, target_sc)
