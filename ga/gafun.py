
from ga.ga import GA
import numpy as np

class GAFun(GA):
    def __init__(self, bp_net, input_number, hidden_number, output_number, popsize, iter_max, PM, PC):
        super(GAFun, self).__init__(bp_net, input_number, hidden_number, output_number, popsize, iter_max, PM, PC)

    def  gafun(self, input_train, output_train, sc):
        gen = 0
        tracematga = np.zeros((self.iter_max, 2))

        chrom = self.genChrome()
        value = self.get_net_predict_value(chrom, input_train, output_train, sc)

        while(gen < self.iter_max):
            #遗传算法选择
            fitnv = self.simple_ranking(value)
            chrom = self.select(chrom, fitnv)
            chrom = self.mutation_ga(chrom)
            chrom = self.crossover(chrom)
            value = self.get_net_predict_value(chrom, input_train, output_train, sc)

            #计算最优
            v1 = min(value)
            index1 = value.index(min(value))
            gen = gen+1
            tracematga[gen, 2] = np.mean(value)

            #记录最优
            if(gen == 1):
                bestChrom_ga = chrom[index1, :]#记录函数1的最优染色体
                bestValue_ga = v1#记录函数1的最优值
            if(bestValue_ga > v1):
                bestChrom_ga = chrom[index1, :]
                bestValue_ga = v1
            tracematga[gen, 1] = bestValue_ga;#保留最优

        return bestChrom_ga, bestValue_ga, tracematga






