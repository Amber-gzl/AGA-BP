from random import random

import numpy as np

from ga.gabp import GABPBase


class GA(GABPBase):
    def __init__(self, bp_net, input_number, hidden_number, output_number, popsize, iter_max, PM, PC):
        super(GA, self).__init__(bp_net, input_number, hidden_number, output_number, popsize, iter_max, PM, PC)

    def sum(self, fit_value):
        total = 0
        for i in range(len(fit_value)):
            total += fit_value[i]
        return total

    def cumsum(self, fit_value):
        for i in range(len(fit_value) - 2, -1, -1):
            t = 0
            j = 0
            while (j <= i):
                t += fit_value[j]
                j += 1
            fit_value[i] = t
            fit_value[len(fit_value) - 1] = 1

    def selection(self, pop, fit_value):
        if len(pop) != self.popsize:
            raise Exception("len of pop error")
        newfit_value = []
        # 适应度总和
        total_fit = sum(fit_value)
        for i in range(len(fit_value)):
            newfit_value.append(fit_value[i] / total_fit)
        # 计算累计概率
        self.cumsum(newfit_value)
        ms = []
        pop_len = len(pop)
        for i in range(pop_len):
            ms.append(random.random())
        ms.sort()
        fitin = 0
        newin = 0
        newpop = pop
        # 转轮盘选择法
        while newin < pop_len:
            if (ms[newin] < newfit_value[fitin]):
                newpop[newin] = pop[fitin]
                newin = newin + 1
            else:
                fitin = fitin + 1
        pop = newpop

    def mutation_ga(self, pop):
        lb = [-1]*self.N
        ub = [1]*self.N
        for i in range(1, self.popsize + 1):
            a = np.random.rand()
            if a < self.PM:
                r = np.random.randint(self.N)
                pop[i, r] = lb(r) + (ub(r) - lb(r)) * np.random.rand()
        return pop

    def crossover(self, pop):
        pnumber = self.N
        index1 = np.random.permutation(self.N)
        long1 = np.fix(self.popsize / 2)
        set1 = index1[0:long1]
        set2 = index1[long1+1:long1*2]
        for i in range(long1):
            a = np.random.rand()
            if a < self.PC:
                index1 = set1(i)
                index2 = set2(i)
                R1 = pop[index1, :]
                R2 = pop[insex2, :]
                r1 = np.random.randint(1, pnumber+1)
                r2 = np.random.randint(1, pnumber+1)
                if(r1 > r2):
                    t1 = r1
                    r1 = r2
                    r2 = t1
                S1 = R1[r1:r2+1]
                S2 = R2[r1:r2+1]
                R1[r1:r2 + 1] = S2
                R2[r1:r2 + 1] = S1
                pop[index1, :] = R1
                pop[index2, :] = R2




