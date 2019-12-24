from .gabp import GABPBase
import numpy as np


class AGABP(GABPBase):
    def __init__(self, bp_net, input_number, hidden_number, output_number, popsize, iter_max, PM, PC,
                 seed=None, PM1=0.05, PM2=0.25, PC1=0.5, PC2=0.8):
        super(AGABP, self).__init__(bp_net, input_number, hidden_number, output_number, popsize, iter_max, PM, PC, seed)
        self.PM1 = PM1  # 变异概率下限
        self.PM2 = PM2  # 变异概率上限
        self.PC1 = PC1  # 交叉概率下限
        self.PC2 = PC2  # 交叉概率上限

    # 目前select固定采用轮盘赌算法, value是get_net_predict_value的返回值
    def ad_select(self, chrom, fit_value, value):
        if len(chrom.shape) != 2:
            raise Exception("ad_select need input param chrom with shape n * m")
        number_sel = max(chrom.shape[0], 2)
        idx = self.rws(fit_value, number_sel)
        sel_chrom = chrom[idx]
        sel_fit_value = fit_value[idx]
        sel_value = value[idx]
        return sel_chrom, sel_fit_value, sel_value

    def ad_mutationGA(self, chrom, fit_value, value, input_test, real_y_test, sc):
        if len(fit_value.shape) != 1:
            raise Exception("ad_mutationGA input fit_value should have one dimension")
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        ub = np.ones((self.N, ))
        lb = -1 * ub
        fmax = np.max(fit_value)
        fmin = np.min(fit_value)
        for i in range(self.popsize):
            a = np.random.rand()
            PM = self.PM2 - (fit_value[i] - fmin) / (fmax - fmin) * (self.PM2 - self.PM1)
            if a < PM:
                r = np.random.randint(self.N)
                chrom[i, r] = lb[r] + (ub[r] - lb[r]) * np.random.rand()
                value[i] = self.get_chroms_predict_value(chrom[i], input_test, real_y_test, sc)[0]

    def ad_crossGA(self, chrom, fit_value, value, input_test, real_y_test, sc):
        index = np.arange(0, self.popsize)
        index1 = np.random.permutation(index)
        long1 = np.fix(self.popsize / 2)
        set1 = index1[0:int(long1)]
        set2 = index1[int(long1):int(long1 * 2)]
        fmax = np.max(fit_value)
        fmin = np.min(fit_value)
        value2 = value.copy()
        for i in range(int(long1)):
            a = np.random.rand()
            PC = self.PC2 - (fit_value[i] - fmin) / (fmax - fmin) * (self.PC2 - self.PC1)
            if a < PC:
                index1 = set1[i]
                index2 = set2[i]
                self.cross_over_once(chrom, index1, index2)
                value2[index1] = self.get_chroms_predict_value(chrom[index1], input_test, real_y_test, sc)
                value2[index2] = self.get_chroms_predict_value(chrom[index2], input_test, real_y_test, sc)
        return value2

    def adgafun(self, input_test, real_y_test, sc):
        gen = 0
        tracematga = np.zeros((self.iter_max, 2))

        chrom = self.genChrome()
        value = self.get_chroms_predict_value(chrom, input_test, real_y_test, sc)
        while (gen < self.iter_max):
            fit_value = self.simple_ranking(value)
            chrom, fit_value2, value2 = self.ad_select(chrom, fit_value, value)
            self.ad_mutationGA(chrom, fit_value2, value2, input_test, real_y_test, sc)
            fit_value2 = self.simple_ranking(value2)
            value = self.ad_crossGA(chrom, fit_value2, value2, input_test, real_y_test, sc)
            index1 = np.argmin(value)
            v1 = value[index1]
            tracematga[gen][1] = np.mean(value)
            if gen == 0:
                bestChrom_ga = chrom[index1, :]  # 记录函数1的最优染色体
                bestValue_ga = v1  # 记录函数1的最优值
            if bestValue_ga > v1:
                bestChrom_ga = chrom[index1, :]
                bestValue_ga = v1
            tracematga[gen][0] = bestValue_ga  # 保留最优
            gen = gen + 1

        return bestChrom_ga, bestValue_ga, tracematga

