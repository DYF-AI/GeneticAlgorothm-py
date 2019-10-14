# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# 初始化种群
def init():
    return np.random.randint(0, 0xFFFFF + 1, size=M)


# 编码转换为决策变量
def B2X(popB, popX):
    for i in range(M):
        x1 = 10.0 * ((popB[i] & 0xFFC00) >> 10) / 1023.0 - 5.0
        x2 = 10.0 * (popB[i] & 0x003FF) / 1023.0 - 5.0
        popX[:, i] = np.array([x1, x2])


# 目标函数定义
def ras(x):
    y = 20 + x[0] ** 2 + x[1] ** 2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))
    return y


# 适应度计算
def getfitness(popB, popX):
    B2X(popB, popX)
    fitness = 1.0 / (ras(popX) + 0.01)
    return fitness


# 依据适应度选择要进行繁殖的个体
def selection(fitness, popB):
    select_probability = fitness / sum(fitness)
    cumulative_sum = np.cumsum(select_probability)
    indexes = np.searchsorted(cumulative_sum, np.random.random(M))
    # resample according to indexes
    popB[:] = popB[indexes]


# 进行基因交叉，实现基因交换
def crossover(popB):
    np.random.shuffle(popB)  # 随机打乱种群中个体的顺序(种群内前一个与后一个配对)
    for i in range(M//2):
        p = np.random.random()  # 随机生成一个0~1内的数
        if p < pc:  # 如果这个数落在交叉概率区间内，则交换部分基因
            index1 = np.random.randint(0, 10)  # 随机选择交叉点
            if (popB[2 * i] & (1 << index1)) != (popB[2 * i + 1] & (1 << index1)):
                if (popB[2 * i] & (1 << index1)):  # 个体1的该位为1(个体2对应位为0)
                    popB[2 * i] &= ~(1 << index1)  # 个体1该位变为0
                    popB[2 * i + 1] |= (1 << index1)  # 个体2该位变为1
                else:  # 个体1的该位为0(个体2对应位为1)
                    popB[2 * i] |= (1 << index1)  # 个体1该位变为1
                    popB[2 * i + 1] &= ~(1 << index1)  # 个体2该位变为0

            index2 = np.random.randint(10, 20)  # 随机选择交叉点
            if (popB[2 * i] & (1 << index2)) != (popB[2 * i + 1] & (1 << index2)):
                if (popB[2 * i] & (1 << index2)):  # 个体1的该位为1(个体2对应位为0)
                    popB[2 * i] &= ~(1 << index2)  # 个体1该位变为0
                    popB[2 * i + 1] |= (1 << index2)  # 个体2该位变为1
                else:  # 个体1的该位为0(个体2对应位为1)
                    popB[2 * i] |= (1 << index2)  # 个体1该位变为1
                    popB[2 * i + 1] &= ~(1 << index2)  # 个体2该位变为0


# 进行基因变异
def mutation(popB):
    for i in range(M):
        p = np.random.random()  # 随机生成一个0~1内的数
        if p < pm:  # 如果这个数落在变异概率区间内，则进行变异处理
            index = np.random.randint(0, 20)  # 采用单点变异
            if (popB[i] & (1 << index)):  # 个体i的该位为1
                popB[i] &= ~(1 << index)  # 个体i的该位变为0
            else:
                popB[i] |= (1 << index)  # 个体i的该位变为1


if __name__ == '__main__':
    M = 50  # 种群大小
    T = 100  # 进化代数
    pc = 0.8  # 交叉概率
    pm = 0.05  # 变异概率

    popB = np.zeros(M, dtype='uint32')  # 由编码表示的种群
    popX = np.zeros((2, M))  # 由决策变量表示的种群
    fitness = np.zeros(M)  # 个体适应度
    fitness_record = np.zeros((2, T))  # 记录适应度随进化代数的变化

    popB = init()  # 随机初始化种群
    t = 0
    while t < T:
        fitness = getfitness(popB, popX)  # 计算适应度
        i = np.argmax(fitness)
        fitness_record[0, t] = sum(fitness) / M
        fitness_record[1, t] = fitness[i]
        selection(fitness, popB)  # 选择
        crossover(popB)  # 交叉
        mutation(popB)  # 变异
        t = t + 1

    max_index = np.argmax(fitness)
    print("X: ", popX[:, max_index])
    print("Y: ", ras(popX[:, max_index]))  # 计算最小值

    plt.plot(np.arange(T), fitness_record[0, :], color='b', label='mean')
    plt.plot(np.arange(T), fitness_record[1, :], color='r', label='best')
    plt.legend(loc='best')
    plt.show()