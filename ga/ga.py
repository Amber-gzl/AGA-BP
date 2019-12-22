from random import random

import numpy as np

from ga.gabp import GABPBase

import math


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
        pop_row = pop.shape[0]
        nsel = max(pop_row, 2)
        selch = []
        fitvsub = fit_value[0:pop_row+1]
        chrix = self.simple_ranking(self, fitvsub, nsel)
        selch = [selch, pop[chrix:]]
        return selch


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
        return pop




