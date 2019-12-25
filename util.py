import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt


def trainBP(input_size, hidden_number, output_size, lr, epochs, shuffle_seed, batch_size, loss_fn,
            x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled):
    torch.manual_seed(shuffle_seed)
    x_train_scaled = torch.from_numpy(x_train_scaled)
    y_train_scaled = torch.from_numpy(y_train_scaled)
    train_dataset = Data.TensorDataset(x_train_scaled, y_train_scaled)
    loader = Data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=1,  # subprocesses for loading data
    )

    net = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_number),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_number, output_size)
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
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
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        # 每个epoch验证一下
        print("epoch ", epoch)
        y_pred = net(x_test_scaled)
        loss = loss_fn(y_pred.detach(), y_test_scaled)
        test_loss.append(loss)
    return net, train_loss, test_loss


def plotBP(train_loss, test_loss):
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


# target_sc for inverse_transform y_pred
def plot_net_predict_result(name, net, x_test_scaled, target_sc, y_test_real):
    x_test_scaled = torch.from_numpy(x_test_scaled)
    y_pred = net(x_test_scaled).detach().numpy()
    y_pred = target_sc.inverse_transform(y_pred)
    plt.plot(y_pred, color='red', label='predict value')
    plt.plot(y_test_real, color='blue', label='real value')
    plt.title(name + 'result')
    plt.ylabel('price')
    plt.xlabel('sample')
    plt.legend()
    plt.show()
