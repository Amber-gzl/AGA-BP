import numpy as np
import torch


class GABPBase(object):
    def __init__(self, bp_net, input_number, hidden_number, output_number, popsize, iter_max, PM, PC):
        self.bp_net = bp_net
        self.bp_net_info = {'input_number': input_number, 'hidden_number': hidden_number,
                            'output_number': output_number}
        self.popsize = popsize  # 遗传算法种群数
        self.iter_max = iter_max  # 遗传算法迭代次数
        self.PM = PM  # 变异概率
        self.PC = PC  # 交叉概率
        self.N = input_number * hidden_number + hidden_number + hidden_number * output_number + output_number

    # 产生种群染色体
    def genChrome(self):
        x = self.__get_flatten_net()
        chrom = np.zeros((self.popsize, self.N))
        for i in range(self.popsize):
            chrom[i] = x + 0.1 * (-0.5 + np.random.rand(1, self.N))
        return chrom

    def recover_net(self, x):
        input_number = self.bp_net_info['input_number']
        hidden_number = self.bp_net_info['hidden_number']
        output_number = self.bp_net_info['output_number']
        w1 = x[:input_number * hidden_number].reshape((hidden_number, input_number))  # 5 * 21 = 105
        b1 = x[input_number * hidden_number: input_number * hidden_number + hidden_number].reshape((hidden_number,))
        w2 = x[(input_number + 1) * hidden_number: (input_number + 1) * hidden_number + hidden_number * output_number].reshape((output_number, hidden_number))
        b2 = x[-output_number:].reshape((output_number,))
        other_net = {'0.weight': torch.from_numpy(w1), '0.bias': torch.from_numpy(b1),
                     '2.weight': torch.from_numpy(w2), '2.bias': torch.from_numpy(b2)}

        net = torch.nn.Sequential(
            torch.nn.Linear(input_number, hidden_number),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_number, output_number)
        )
        net.load_state_dict(other_net)
        print(net.state_dict())
        return net

    def get_net_predict_value(self, chrom, input_test, real_y_test, sc):
        input_test = torch.from_numpy(input_test)
        value = []
        for i in range(self.popsize):
            tmp_net = self.recover_net(chrom[i])
            y_pred = tmp_net(input_test)
            y_pred = sc.inverse_transform(y_pred.detach().numpy())
            value.append(np.sum(np.square(y_pred - real_y_test)))
        return value

    def __get_flatten_net(self):
        state_dict = self.bp_net.state_dict()
        w1 = state_dict['0.weight'].numpy().reshape((1, -1))
        b1 = state_dict['0.bias'].numpy().reshape((1, -1))
        x = np.hstack((w1, b1))
        w2 = state_dict['2.weight'].numpy().reshape((1, -1))
        x = np.hstack((x, w2))
        b2 = state_dict['2.bias'].numpy().reshape((1, -1))
        x = np.hstack((x, b2))
        return x
