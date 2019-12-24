from random import random

import numpy as np

from ga.gabp import GABPBase

import math


class GA(GABPBase):
    def __init__(self, bp_net, input_number, hidden_number, output_number, popsize, iter_max, PM, PC):
        super(GA, self).__init__(bp_net, input_number, hidden_number, output_number, popsize, iter_max, PM, PC)


    def select(self, pop, fit_value):
        pop_row = pop.shape[0]
        nsel = max(pop_row, 2)
        chrix = self.rws(fit_value, nsel)
        selch = pop[chrix]
        return selch


    def mutation_ga(self, pop):
        lb = [-1]*self.N
        ub = [1]*self.N
        for i in range(0, self.popsize):
            a = np.random.rand()
            if a < self.PM:
                r = np.random.randint(self.N)
                pop[i][r] = lb[r] + (ub[r] - lb[r]) * np.random.rand()
        return pop

    def crossover(self, pop):
        pnumber = self.N
        index = np.arange(0, self.popsize)
        index1 = np.random.permutation(index)
        long1 = np.fix(self.popsize / 2)
        set1 = index1[0:int(long1)]
        set2 = index1[int(long1):int(long1*2)]
        for i in range(int(long1)):
            a = np.random.rand()
            if a < self.PC:
                index1 = set1[i]
                index2 = set2[i]
                R1 = pop[index1]
                R2 = pop[index1]
                r1 = np.random.randint(0, pnumber)
                r2 = np.random.randint(0, pnumber)
                if(r1 > r2):
                    t1 = r1
                    r1 = r2
                    r2 = t1
                S1 = R1[r1:r2+1]
                S2 = R2[r1:r2+1]
                R1[r1:r2 + 1] = S2
                R2[r1:r2 + 1] = S1
                pop[index1] = R1
                pop[index2] = R2
        return pop

    def  gafun(self, input_train, output_train, sc):
        gen = 0
        tracematga = np.zeros((self.iter_max, 2))

        chrom = self.genChrome()
        value = self.get_chroms_predict_value(chrom, input_train, output_train, sc)

        while(gen < self.iter_max):
            #遗传算法选择
            fitnv = self.simple_ranking(value)
            chrom = self.select(chrom, fitnv)
            chrom = self.mutation_ga(chrom)
            chrom = self.crossover(chrom)
            value = self.get_chroms_predict_value(chrom, input_train, output_train, sc)

            #计算最优
            v1 = min(value)
            index1 = np.where(value == max(value))
            tracematga[gen][1] = np.mean(value)

            #记录最优
            if(gen == 0):
                bestChrom_ga = chrom[index1]#记录函数1的最优染色体
                bestValue_ga = v1#记录函数1的最优值
            if(bestValue_ga > v1):
                bestChrom_ga = chrom[index1]
                bestValue_ga = v1
            tracematga[gen][0] = bestValue_ga;#保留最优
            gen = gen + 1

        return bestChrom_ga, bestValue_ga, tracematga




