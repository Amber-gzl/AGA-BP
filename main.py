import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    csv_data = pd.read_csv("./累计成交额含价格.csv", header=None)

    # 数据预处理
    x = csv_data.iloc[:, 1:-1].values
    Y = csv_data.iloc[:, -1].values
    x = x.astype(np.float32)
    Y = Y.astype(np.float32)
    print(x.shape)
    print(Y.shape)
    print(x.dtype)
    print(Y.dtype)

    # ----------------------------BP网络超参数----------------------------
    shuffle_seed = 233
    test_set_size = 10
    hidden_number = 5  # 仅有一层隐藏层，神经元个数为5
    learning_rate = 2e-2
    epochs = 1000
    BATCH_SIZE = 4

    # 获取训练集和测试集
    x_train = x[:-test_set_size]
    x_test = x[-test_set_size:]
    Y_train = Y[:-test_set_size]
    Y_test = Y[-test_set_size:]

    # MinMax归一化
    sc = MinMaxScaler(feature_range=(-1, 1))
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)

    # These line is for debug! Should be the same with matlab version.
    print(x_train[0])
    print(x_train_scaled[0])
    print(x_train_scaled[162])

    torch.manual_seed(shuffle_seed)
    x_train_scaled = torch.from_numpy(x_train_scaled)
    Y_train = torch.from_numpy(Y_train)
    train_dataset = Data.TensorDataset(x_train_scaled, Y_train)
    loader = Data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=1,  # subprocesses for loading data
    )

    input_size = x_test_scaled.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_number),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_number, 1)
    )

    mynet = {}
    mynet['0.weight'] = torch.randn((5, 21), dtype=torch.float32)
    mynet['0.bias'] = torch.randn(5, dtype=torch.float32)
    mynet['2.weight'] = torch.randn((1, 5), dtype=torch.float32)
    mynet['2.bias'] = torch.randn(1, dtype=torch.float32)
    net.load_state_dict(mynet)
    print(net.state_dict())
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            # print(batch_x)
            # print(batch_x.dtype)
            optimizer.zero_grad()
            y_pred = net(batch_x)
            loss = loss_fn(y_pred, batch_y)
            # print("epoch: " + str(epoch) + " step: " + str(step) + "loss:" + str(loss.data))

            loss.backward()
            optimizer.step()

    print(net.state_dict())
