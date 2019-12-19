import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
    learning_rate = 2e-2
    epochs = 50
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

    torch.manual_seed(shuffle_seed)
    x_train_scaled = torch.from_numpy(x_train_scaled)
    y_train_scaled = torch.from_numpy(y_train_scaled)
    train_dataset = Data.TensorDataset(x_train_scaled, y_train_scaled)
    loader = Data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=1,  # subprocesses for loading data
    )

    input_size = x_train_scaled.shape[1]
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
    # 测试集非常小，可以一次放入进行测试，当测试集增大，这里也需要修改为和train相同的dataloader的形式
    x_test_scaled = torch.from_numpy(x_test_scaled)
    y_test_scaled = torch.from_numpy(y_test_scaled)
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            y_pred = net(batch_x)
            loss = loss_fn(y_pred, batch_y)
            train_loss.append(loss.item() / float(BATCH_SIZE))
            # print("epoch: " + str(epoch) + " step: " + str(step) + "loss:" + str(loss.data))

            loss.backward()
            optimizer.step()
        # 每个epoch验证一下
        print("epoch ", epoch)
        y_pred = net(x_test_scaled)
        loss = loss_fn(y_pred.detach(), y_test_scaled)
        test_loss.append(loss)

    plt.plot(train_loss, color='red', label='train loss')
    plt.title('Train Loss')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(test_loss, color='red', label='test loss')
    plt.title('Test Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    y_pred = net(x_test_scaled).detach().numpy()
    y_pred = target_sc.inverse_transform(y_pred)
    plt.plot(y_pred, color='red', label='predict value')
    plt.plot(y_test, color='blue', label='real value')
    plt.title('result')
    plt.ylabel('price')
    plt.xlabel('sample')
    plt.legend()
    plt.show()

    print(net.state_dict())
