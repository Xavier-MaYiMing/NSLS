#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/13 17:44
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : NSLS.py
# @Statement : Nondominated sorting and local search (NSLS) based algorithm for multi-objective optimization
# @Reference : Chen B, Zeng W, Lin Y, et al. A new local search-based multiobjective optimization algorithm[J]. IEEE Transactions on Evolutionary Computation, 2014, 19(1): 50-73.
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def cal_obj(x):
    # ZDT3
    if np.any(x < 0) or np.any(x > 1):
        return [np.inf, np.inf]
    f1 = x[0]
    num1 = 0
    for i in range(1, len(x)):
        num1 += x[i]
    g = 1 + 9 * num1 / (len(x) - 1)
    f2 = g * (1 - np.sqrt(x[0] / g) - x[0] / g * np.sin(10 * np.pi * x[0]))
    return [f1, f2]


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 0
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def nd_fc_sort(pop, objs, npop):
    # sort the population according to the Pareto rank and farthest candidate method
    new_pop = np.zeros((npop, pop.shape[1]))
    new_objs = np.zeros((npop, objs.shape[1]))
    pfs, rank = nd_sort(objs)
    ind = num = 0
    while num + len(pfs[ind]) <= npop:
        new_pop[num: num + len(pfs[ind])] = pop[pfs[ind]]
        new_objs[num: num + len(pfs[ind])] = objs[pfs[ind]]
        num += len(pfs[ind])
        ind += 1
    # choose K solutions from a population F
    F = pop[pfs[ind]]
    F_objs = objs[pfs[ind]]
    K = npop - num
    p_accept = np.full(F.shape[0], False)
    for i in range(objs.shape[1]):
        p_accept[np.argmin(F_objs[:, i])] = True
        p_accept[np.argmax(F_objs[:, i])] = True
    if np.sum(p_accept) > K:
        accepted = np.where(p_accept)[0]
        p_accept[accepted[np.random.choice(np.sum(p_accept), np.sum(p_accept) - K, replace=False)]] = False
    elif np.sum(p_accept) < K:
        dis = squareform(pdist(F_objs, metric='euclidean'), force='no', checks=True)
        eye = np.arange(F.shape[0])
        dis[eye, eye] = np.inf
        while np.sum(p_accept) < K:
            remain = np.where(~p_accept)[0]
            best = np.argmax(np.min(dis[~p_accept][:, p_accept], axis=1))
            p_accept[remain[best]] = True
    new_pop[num:] = F[p_accept]
    new_objs[num:] = F_objs[p_accept]
    return new_pop, new_objs


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def main(npop, iter, lb, ub, mu=0.5, delta=0.1):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param mu: the mean value of the Gaussian distribution (default = 0.5)
    :param delta: the standard deviation of the Gaussian distribution (default = 0.1)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = np.array([cal_obj(x) for x in pop])  # objectives

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 20 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        # Step 2.1. Local search
        new_pop = pop.copy()
        new_objs = objs.copy()
        for i in range(npop):
            for j in range(nvar):
                c = mu + delta * np.random.randn()
                [r1, r2] = np.random.choice(npop, 2, replace=False)
                new_ind1 = new_pop[i].copy()
                new_ind2 = new_pop[i].copy()
                new_ind1[j] += c * (pop[r1, j] - pop[r2, j])
                new_ind1[j] = max(new_ind1[j], lb[j])
                new_ind1[j] = min(new_ind1[j], ub[j])
                new_ind2[j] -= c * (pop[r1, j] - pop[r2, j])
                new_ind2[j] = max(new_ind2[j], lb[j])
                new_ind2[j] = min(new_ind2[j], ub[j])
                new_obj1 = cal_obj(new_ind1)
                new_obj2 = cal_obj(new_ind2)
                flag1 = np.sum([np.any(new_obj1 < objs[i]), ~np.any(new_obj1 > objs[i])])
                flag2 = np.sum([np.any(new_obj2 < objs[i]), ~np.any(new_obj2 > objs[i])])
                if flag1 == 0 and flag2 == 0:
                    continue
                elif flag1 > flag2:
                    new_pop[i] = new_ind1
                    new_objs[i] = new_obj1
                elif flag1 < flag2:
                    new_pop[i] = new_ind2
                    new_objs[i] = new_obj2
                elif np.random.random() < 0.5:
                    new_pop[i] = new_ind1
                    new_objs[i] = new_obj1
                else:
                    new_pop[i] = new_ind2
                    new_objs[i] = new_obj2

        # Step 2.2. Environmental selection
        pop, objs = nd_fc_sort(np.concatenate((pop, new_pop), axis=0), np.concatenate((objs, new_objs), axis=0), npop)

    # Step 3. Sort the results
    dom = np.full(npop, False)
    for i in range(npop - 1):
        for j in range(i, npop):
            if not dom[i] and dominates(objs[j], objs[i]):
                dom[i] = True
            if not dom[j] and dominates(objs[i], objs[j]):
                dom[j] = True
    pf = objs[~dom]
    plt.figure()
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    plt.scatter(x, y)
    plt.xlabel('objective 1')
    plt.ylabel('objective 2')
    plt.title('The Pareto front of ZDT3')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(100, 250, np.array([0] * 30), np.array([1] * 30))
