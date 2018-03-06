import xlrd
import numpy as np
from Polynomial import Polynomial


# 读取excel表
def read_excel(file_name):
    excel_file = xlrd.open_workbook(file_name)
    print(excel_file.sheet_names())
    use_sheet = excel_file.sheet_by_name("Sheet2")
    print(use_sheet.name, use_sheet.nrows, use_sheet.ncols)
    x_value = use_sheet.col_values(0)
    y_value = use_sheet.col_values(1)
    x_value = np.array(x_value[1:], np.float)
    y_value = np.array(y_value[1:], np.float)
    print(x_value)
    print(y_value)
    return x_value, y_value


# 根据原始x，y序列构造新的X，Y序列
def construct_profile(value):
    mean_value = np.mean(value)
    print("mean value: %f" % mean_value)

    profile = []

    for i in range(len(value)):
        temp_sum = 0
        for j in range(i+1):
            temp_sum += (value[j] - mean_value)
        profile.append(temp_sum)
    print("profile's length: %d" % len(profile))

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


# 梯度下降，返回次数项由低到高对应的系数，包括0次项
def gradient_descent(x, y, m=4, eta=0.001, max_iterations=10000):
    x_t = x.transpose()
    parameters = np.zeros(m+1)  # 初始化参数
    length = len(x)
    for i in range(max_iterations):
        hypothesis = np.dot(x, parameters.transpose())
        loss = hypothesis - y
        gradient = np.dot(x_t, loss.transpose()) / length  # 计算梯度
        parameters = parameters - eta * gradient  # 更新参数
        if (i+1) % 100 == 0:
            cost = np.sum(loss ** 2) / (2 * length)
            print("Iteration %d | Cost: %f" % (i+1, cost))
    return parameters


# 拟合多项式
def fit_polynomial(division, s, m, eta, max_iterations):
    x, y = construct_xy(division, m)
    parameters = gradient_descent(x, y, m=m, eta=eta, max_iterations=max_iterations)
    new_polynomial = Polynomial(parameters)
    return new_polynomial


# 由分割序列构造拟合值x，y
def construct_xy(division, m):
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


if __name__ == "__main__":
    x_value, y_value = read_excel("价量.xlsx")

    x_profile = construct_profile(x_value)
    y_profile = construct_profile(y_value)

    s = 50
    m = 4
    eta = 2e-13
    # m = 2
    # eta = 0.0001
    max_iterations = 1000000
    x_division = divide_profile(x_profile, s)
    y_division = divide_profile(y_profile, s)

    x_polynomial = fit_polynomial(x_division, s, m, eta, max_iterations)

