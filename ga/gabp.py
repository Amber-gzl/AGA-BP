import numpy as np
import torch


class GABPBase(object):
    def __init__(self, bp_net, input_number, hidden_number, output_number, popsize, iter_max, PM, PC, seed=None):
        self.bp_net = bp_net
        self.bp_net_info = {'input_number': input_number, 'hidden_number': hidden_number,
                            'output_number': output_number}
        self.popsize = popsize  # 遗传算法种群数
        self.iter_max = iter_max  # 遗传算法迭代次数
        self.PM = PM  # 变异概率
        self.PC = PC  # 交叉概率
        self.N = input_number * hidden_number + hidden_number + hidden_number * output_number + output_number
        self.random_seed = seed

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
        w2 = x[(input_number + 1) * hidden_number: (
                                                           input_number + 1) * hidden_number + hidden_number * output_number].reshape(
            (output_number, hidden_number))
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

    # 输入参数是染色体，测试集输入以及真实样本值，同时还有用于还原预测值的MinMaxScaler
    # 输出是一个numpy.array，行向量，保存每一个染色体的误差，用于计算适应度
    def get_chroms_predict_value(self, chrom, input_test, real_y_test, sc):
        input_test = torch.from_numpy(input_test)
        value = []
        # chrom的长度不需要和popsize相同，这个函数只是根据传入的chrom返回其误差
        for i in range(chrom.shape[0]):
            tmp_net = self.recover_net(chrom[i])
            y_pred = tmp_net(input_test)
            y_pred = sc.inverse_transform(y_pred.detach().numpy())
            value.append(np.sum(np.square(y_pred - real_y_test)))
        return np.array(value, dtype=np.float32)

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

    # need test! input value [i for in range(100)]; [10 for _ in range(100)]
    def simple_ranking(self, value):
        if value.dtype != np.float32:
            raise Exception("传入参数类型不是np.float32类型")
        if len(value) != self.popsize:
            raise NotImplementedError("目前还没有实现这种情况的处理")
        # 走到这里预计value已经被正确处理了，里面的缺省值应该是某些预设的值，否则value的类型不会是np.float32，但缺省值不正确会导致程序难以调试
        ix = np.argsort(-value)
        sorted_value = value[ix]
        helpper = np.array([i for i in range(self.popsize)], dtype=np.float32)
        helpper = 2 * helpper / (self.popsize - 1)
        start = 0
        end = 1
        fit_value = None
        while end < self.popsize:
            if np.equal(sorted_value[start], sorted_value[end]):
                end += 1
                continue
            # sorted_value[start] != sorted_value[end] [start, end), end没有被处理，所以start = end
            tmp_result = np.sum(helpper[start:end]) * np.ones(end - start, dtype=np.float32) / (end - start)
            if fit_value is None:
                fit_value = tmp_result
            else:
                fit_value = np.hstack((fit_value, tmp_result))
            start = end
            end = start + 1

        if not (start < self.popsize and end == self.popsize):
            raise Exception("没想到吧?")

        tmp_result = np.sum(helpper[start:end]) * np.ones(end - start, dtype=np.float32) / (end - start)
        if fit_value is None:
            fit_value = tmp_result
        else:
            fit_value = np.hstack((fit_value, tmp_result))
        # Finally, return unsorted vector.
        uix = np.argsort(ix)
        fit_value = fit_value[uix]
        return fit_value

    # 实现轮盘赌算法, 返回指针, 要求fit_value是一个numpy一维数组, 返回值是索引
    def rws(self, fit_value, number_select):
        if fit_value.dtype != np.float32:
            raise Exception("rws input fit_value is not np.float32")
        if len(fit_value.shape) != 1:
            raise Exception("rws input fit_value is not one dimension")
        fit_value_length = fit_value.shape[0]
        cumfit = np.cumsum(fit_value)  # 行向量
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        trials = cumfit[fit_value_length - 1] * np.random.rand(number_select, 1)
        cumfit = cumfit.reshape((1, fit_value_length))
        helper = np.hstack((np.zeros((number_select, 1)), cumfit[[0] * number_select][:, 0:-1]))
        return np.argwhere(np.logical_and(cumfit > trials, helper <= trials))[:, -1]

