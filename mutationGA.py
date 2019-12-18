import random


def mutation_ga(chrom, popsize, pm, n, lb, ub):
    for i in range(1, popsize+1):
        a = random.random()
        if a < pm:
            r = random.randint(n)
            chrom[i, r] = lb(r) + (ub(r) - lb(r)) * random.random()
    return chrom
