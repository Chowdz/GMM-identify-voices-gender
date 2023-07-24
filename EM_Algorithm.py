"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2022/10/14 10:11 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random

global pp1, pp2


# 二维高斯分布密度
def PDF(u, sigma2, p, x):
    part1 = 1 / (2 * np.pi * np.power(sigma2[0], 0.5) * np.power(sigma2[1], 0.5) * np.power(1 - p ** 2, 0.5))
    part2 = - 1 / (2 * (1 - p ** 2))
    part3 = (np.power(x[0] - u[0], 2)) / sigma2[0] - (2 * p * (x[0] - u[0]) * (x[1] - u[1])) / (
            np.power(sigma2[0], 0.5) * np.power(sigma2[1], 0.5)) + (np.power(x[1] - u[1], 2)) / sigma2[1]
    if part1 * np.exp(part2 * part3) < 2.2250738585072014e-308:
        pdf = 2.2250738585072014e-300
    else:
        pdf = part1 * np.exp(part2 * part3)
    return pdf


# 带有隐变量的似然函数值
def Likelihood(data, z, u, sigma2, p):
    data = pd.DataFrame(data)
    number = len(data.index.values)
    likelihood_list = []
    for i in range(number):
        pp_1, pp_2 = E(z, u, sigma2, p, data.iloc[i, :])
        likelihood_single = pp_1 * PDF(u[:2], sigma2[:2], p[0], data.iloc[i, :]) + pp_2 * PDF(u[2:], sigma2[2:], p[1],
                                                                                              data.iloc[i, :])
        likelihood_list.append(likelihood_single)
    likelihood = sum(likelihood_list)
    likelihood_list.clear()
    return likelihood


# E步
def E(z, u, sigma2, p, x):
    """
    :param x: 某个样本向量，在这里是二维的，因为是二维正态分布
    :param z: 隐变量概率值向量
    :param u: 均值向量，前两个是第一个分布的，后两个是第二个分布的
    :param sigma2: 方差向量，前两个是第一个分布的，后两个是第二个分布的
    :param p: 相关系数向量，前一个是第一个分布的，后一个是第二个分布的
    :return: 在样本和未知参数给定的条件下，属于该隐变量的条件概率
    """
    prior_probability1 = z[0] * PDF(u[:2], sigma2[:2], p[0], x)
    prior_probability2 = z[1] * PDF(u[2:], sigma2[2:], p[1], x)
    posterior_probability1 = prior_probability1 / (prior_probability1 + prior_probability2)
    return posterior_probability1, 1 - posterior_probability1


# M步
def M(data, z, u, sigma2, p, N=10000, epsilon=1e-7):
    global pp1, pp2
    data = pd.DataFrame(data)
    number = len(data.index.values)
    n_iter = 0
    while n_iter <= N:
        print(f"迭代第{n_iter + 1}次")
        last_z, last_u, last_sigma2, last_p = z, u, sigma2, p
        pp1_sum, pp2_sum = [], []
        pp1_1_sum, pp1_2_sum, pp2_1_sum, pp2_2_sum = [], [], [], []
        pp1_1_s_sum, pp1_2_s_sum, pp2_1_s_sum, pp2_2_s_sum = [], [], [], []
        for j in range(number):
            pp1, pp2 = E(z, u, sigma2, p, data.iloc[j, :])
            pp1_1, pp1_2 = pp1 * data.iloc[j, 0], pp1 * data.iloc[j, 1]
            pp2_1, pp2_2 = pp2 * data.iloc[j, 0], pp2 * data.iloc[j, 1]
            pp1_sum.append(pp1)
            pp2_sum.append(pp2)
            pp1_1_sum.append(pp1_1)
            pp1_2_sum.append(pp1_2)
            pp2_1_sum.append(pp2_1)
            pp2_2_sum.append(pp2_2)
        z1, z2 = sum(pp1_sum) / number, sum(pp2_sum) / number
        u11, u12, u21, u22 = sum(pp1_1_sum) / sum(pp1_sum), sum(pp1_2_sum) / sum(pp1_sum), sum(pp2_1_sum) / sum(
            pp2_sum), sum(pp2_2_sum) / sum(pp2_sum)
        for k in range(number):
            pp1, pp2 = E(z, u, sigma2, p, data.iloc[k, :])
            pp1_1_s, pp1_2_s = pp1 * np.power(data.iloc[k, 0] - u11, 2), pp1 * np.power(data.iloc[k, 1] - u12, 2)
            pp2_1_s, pp2_2_s = pp2 * np.power(data.iloc[k, 0] - u21, 2), pp2 * np.power(data.iloc[k, 1] - u22, 2)
            pp1_1_s_sum.append(pp1_1_s)
            pp1_2_s_sum.append(pp1_2_s)
            pp2_1_s_sum.append(pp2_1_s)
            pp2_2_s_sum.append(pp2_2_s)
        sigma2_11 = sum(pp1_1_s_sum) / sum(pp1_sum)
        sigma2_12 = sum(pp1_2_s_sum) / sum(pp1_sum)
        sigma2_21 = sum(pp2_1_s_sum) / sum(pp2_sum)
        sigma2_22 = sum(pp2_2_s_sum) / sum(pp2_sum)
        z = [z1, z2]
        u = [u11, u12, u21, u22]
        sigma2 = [sigma2_11, sigma2_12, sigma2_21, sigma2_22]
        p = [z1 * pd.Series(data.iloc[:, 1]).corr(pd.Series(data.iloc[:, 0]), method='pearson'),
             z2 * pd.Series(data.iloc[:, 1]).corr(pd.Series(data.iloc[:, 0]), method='pearson')]
        print(f"z的值为：{z}")
        print(f"u的值为：{u}")
        print(f"sigma2的值为：{sigma2}")
        print(f"p的值为：{p}")
        pp1_sum.clear()
        pp2_sum.clear()
        pp1_1_sum.clear()
        pp1_2_sum.clear()
        pp2_1_sum.clear()
        pp2_2_sum.clear()
        pp1_1_s_sum.clear()
        pp1_2_s_sum.clear()
        pp2_1_s_sum.clear()
        pp2_2_s_sum.clear()
        last_likelihood = Likelihood(data, last_z, last_u, last_sigma2, last_p)
        new_likelihood = Likelihood(data, z, u, sigma2, p)
        print(f"两次似然函数差值：{np.abs(new_likelihood - last_likelihood)}\n")
        if np.abs(new_likelihood - last_likelihood) < epsilon:
            break
        n_iter += 1
    pd.DataFrame(data={'u': u, 'sigma2': sigma2}).to_csv('feature_ult.csv')
    pd.DataFrame(data={'p': p}).to_csv('p_ult.csv')
    return z, u, sigma2, p


# 画图函数
def plot_gaussian(mean_sigma, p):
    reference_data = pd.DataFrame(pd.read_csv(mean_sigma))
    p_data = pd.DataFrame(pd.read_csv(p))
    u_list = list(reference_data.iloc[:, 0])
    sigma_list = list(reference_data.iloc[:, 1])
    p_list = list(p_data.iloc[:, 0])
    mean1 = [u_list[0], u_list[1]]
    mean2 = [u_list[2], u_list[3]]
    cov1 = [[sigma_list[0], p_list[0] * np.power(sigma_list[0], 0.5) * np.power(sigma_list[1], 0.5)],
            [p_list[0] * np.power(sigma_list[0], 0.5) * np.power(sigma_list[1], 0.5), sigma_list[1]]]
    cov2 = [[sigma_list[2], p_list[1] * np.power(sigma_list[2], 0.5) * np.power(sigma_list[3], 0.5)],
            [p_list[1] * np.power(sigma_list[2], 0.5) * np.power(sigma_list[3], 0.5), sigma_list[3]]]
    Gaussian1 = multivariate_normal(mean=mean1, cov=cov1)
    Gaussian2 = multivariate_normal(mean=mean2, cov=cov2)
    X, Y = np.meshgrid(np.linspace(25, 170, 50), np.linspace(-5, 15, 50))
    d = np.dstack([X, Y])
    Z1 = Gaussian1.pdf(d).reshape(50, 50)
    Z2 = Gaussian2.pdf(d).reshape(50, 50)
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='rainbow', alpha=0.6)
    ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap='rainbow', alpha=0.6)
    ax.set_xlabel('F0')
    ax.set_ylabel('IQR')
    ax.set_zlabel('pdf')
    plt.show()
    return


if __name__ == '__main__':
    voice_data = pd.read_csv('transient1.csv')
    z_init = [0.6, 0.4]
    u_init = [129, 0.11, 69, 5.7]
    sigma2_init = [25, 0.0035, 2.45, 3.78]
    p_init = [0.5, 0.5]
    # z_ult, u_ult, sigma2_ult, p_ult = M(voice_data, z_init, u_init, sigma2_init, p_init)
    plot_gaussian('feature_ult.csv', 'p_ult.csv')
