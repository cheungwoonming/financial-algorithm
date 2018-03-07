import xlrd
import numpy as np
from scipy.optimize import leastsq
import pylab as pl


# 读取excel表
def read_excel(file_name):
    excel_file = xlrd.open_workbook(file_name)
    # print(excel_file.sheet_names())
    use_sheet = excel_file.sheet_by_name("Sheet2")
    # print(use_sheet.name, use_sheet.nrows, use_sheet.ncols)
    x_value = use_sheet.col_values(0)
    y_value = use_sheet.col_values(1)
    x_value = np.array(x_value[1:], np.float)
    y_value = np.array(y_value[1:], np.float)
    # print(x_value)
    # print(y_value)
    return x_value, y_value


# 根据原始x，y序列构造新的X，Y序列
def construct_profile(value):
    mean_value = np.mean(value)
    print("原序列均值：%f" % mean_value)
    profile = []

    for i in range(len(value)):
        temp_sum = 0
        for j in range(i+1):
            temp_sum += (value[j] - mean_value)
        profile.append(temp_sum)
    print("新序列均值：%f， 方差：%f" % (np.mean(profile), np.std(profile)))

    return profile


# 以长度s分割序列为2Ns段
def divide_profile(profile, s):
    if s < 10 or s > len(x_profile)/5:
        print("s's length error!")
    N = len(profile)
    Ns = int(N/s)
    division = []
    for i in range(Ns):
        division.append(profile[i*s:(i+1)*s])
    for i in range(Ns):
        division.append(profile[N-s*(i+1): N-s*i])
    return division


# 由分割序列构造拟合值x，y
def construct_xy2(division, m):
    n_row = np.shape(division)[0]
    n_column = np.shape(division)[1]
    x = np.zeros(shape=(n_row*n_column, m+1))  # (2Ns*s = row*column, m+1)
    y = np.zeros(shape=(n_row*n_column))
    for i in range(n_row):
        for j in range(n_column):
            for k in range(m+1):
                x[i * n_column + j][k] = (j+1) ** k  # i ** K
            y[i * n_column + j] = division[i][j]
    return x, y


# 梯度下降，返回次数项由低到高对应的系数，包括0次项
def gradient_descent(x, y, m=4, eta=0.001, max_iterations=10000):
    x_t = x.transpose()
    parameters = np.zeros(m+1)  # 初始化参数
    # parameters = np.array([-1.55245949e+01, 1.97244256e-02, 3.59871120e-03, -1.83936582e-04, 2.15314155e-06])
    length = len(x)
    for i in range(max_iterations):
        hypothesis = np.dot(x, parameters.transpose())
        loss = hypothesis - y
        gradient = np.dot(x_t, loss.transpose()) / length  # 计算梯度
        parameters = parameters - eta * gradient  # 更新参数
        if (i+1) % 1000 == 0:
            cost = np.sum(loss ** 2) / (2 * length)
            print("Iteration %d | Cost: %f" % (i+1, cost))
    return parameters


# 拟合多项式
def fit_polynomial2(division, s, m, eta, max_iterations):
    # 自己写的梯度下降，仅在低维度的情况下可用
    x, y = construct_xy(division, m)
    parameters = gradient_descent(x, y, m=m, eta=eta, max_iterations=max_iterations)
    print(parameters)
    return parameters


# 多项式函数
def polynomial_func(p, x):
    f = np.poly1d(p)
    return f(x)


# 残差函数
def loss_func(p, y, x):
    ret = polynomial_func(p, x) - y
    return ret


def construct_xy(division):
    n_row = np.shape(division)[0]
    n_column = np.shape(division)[1]
    x = np.zeros(shape=(n_row * n_column))
    y = np.zeros(shape=(n_row * n_column))
    for i in range(n_row):
        for j in range(n_column):
            x[i * n_column + j] = (j + 1)
            y[i * n_column + j] = division[i][j]
    return x, y


# 拟合多项式
def fit_polynomial(division, m):
    x, y = construct_xy(division)
    p_init = np.zeros(m+1)
    # 最小二乘拟合
    plsq = leastsq(loss_func, p_init, args=(y, x))
    parameters = plsq[0]

    cost = np.sum(loss_func(parameters, y, x) ** 2) / len(x)
    print("RMSE: %f" % (cost ** 0.5))
    print("parameters : %s" % parameters)

    return np.poly1d(parameters)


def compute_covariance(x_func, x_segments, y_func, y_segments):
    length = np.shape(x_segments)[0]
    s = np.shape(x_segments)[1]

    covariance = []
    i_array = np.linspace(1, s, s)

    for v in range(length):
        temp = abs(x_segments[v] - x_func(i_array)) * abs(y_segments[v] - y_func(i_array))
        covariance.append(np.sum(temp)/s)

    return np.array(covariance)


def fluctuation_func(array, q):
    if q != 0:
        return np.power(np.mean(np.power(array, q/2)), 1/q)
    else:
        return np.exp(np.mean(np.log(array))/2)


if __name__ == "__main__":
    s = 20
    m = 5
    q = 10

    x_value, y_value = read_excel("价量.xlsx")

    # 得到新序列
    x_profile = construct_profile(x_value)
    y_profile = construct_profile(y_value)

    temp_array = np.arange(2.75, 5.6, 0.25)
    s_array = [int(np.exp(i)) for i in temp_array]
    result = []
    for s in s_array:
        # 得到分割后的（2Ns * s）序列
        x_division = divide_profile(x_profile, s)
        y_division = divide_profile(y_profile, s)

        # 得到多项式拟合函数
        x_polynomial_func = fit_polynomial(x_division, m)
        y_polynomial_func = fit_polynomial(y_division, m)

        # 得到去势协方差
        F2 = compute_covariance(x_polynomial_func, x_division, y_polynomial_func, y_division)

        Fqs = []
        q_array = [i for i in range(-q, q+1, 2)]
        for i in q_array:
            Fqs.append(fluctuation_func(F2, i))
        print(Fqs)
        result.append(Fqs)

    result = np.array(result)
    result = result.transpose()
    s_array = np.log(s_array)
    result = np.log(result)
    for row in result:
        pl.plot(s_array, row)
    pl.xlabel("lns")
    pl.ylabel("lnFq")
    pl.show()
