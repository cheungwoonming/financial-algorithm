import xlrd
import numpy as np
from scipy.optimize import leastsq
import pylab as pl
import csv

# m值
m = 4
# q值的最大值，以2位间隔计算[-q, q]的值
q = 10


# 读取excel表
def read_excel(file_name):
    excel_file = xlrd.open_workbook(file_name)
    use_sheet = excel_file.sheet_by_name("Sheet2")
    x_value = use_sheet.col_values(0)
    y_value = use_sheet.col_values(1)
    x_value = np.array(x_value[1:], np.float)
    y_value = np.array(y_value[1:], np.float)
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


# 不用了
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
        if (i+1) % 1000 == 0:
            cost = np.sum(loss ** 2) / (2 * length)
            print("Iteration %d | Cost: %f" % (i+1, cost))
    return parameters


# 不用了
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
    x_array = np.zeros(shape=(n_row, n_column))
    y_array = np.zeros(shape=(n_row, n_column))
    for i in range(n_row):
        for j in range(n_column):
            x_array[i][j] = (j + 1)
            y_array[i][j] = division[i][j]
    return x_array, y_array


# 拟合多项式
def fit_polynomial(division, m):
    x_array, y_array = construct_xy(division)
    p_init = np.zeros(m+1)
    polynomial_func_array = []
    RMSE = []
    for i in range(len(x_array)):
        # 最小二乘拟合
        plsq = leastsq(loss_func, p_init, args=(y_array[i], x_array[i]))
        parameters = plsq[0]

        cost = np.sum(loss_func(parameters, y_array[i], x_array[i]) ** 2) / len(x_array[i])
        RMSE.append(cost ** 0.5)
        polynomial_func_array.append(np.poly1d(parameters))
    print(RMSE)
    return polynomial_func_array, RMSE


def compute_covariance(x_func_array, x_segments, y_func_array, y_segments):
    length = np.shape(x_segments)[0]
    s = np.shape(x_segments)[1]

    covariance = []
    i_array = np.linspace(1, s, s)

    for v in range(length):
        temp = abs(x_segments[v] - x_func_array[v](i_array)) * abs(y_segments[v] - y_func_array[v](i_array))
        covariance.append(np.mean(temp))

    return np.array(covariance)


def fluctuation_func(array, q):
    if q != 0:
        return np.power(np.mean(np.power(array, q/2)), 1/q)
    else:
        return np.exp(np.mean(np.log(array))/2)


if __name__ == "__main__":
    x_value, y_value = read_excel("data/价量.xlsx")

    # 得到新序列
    x_profile = construct_profile(x_value)
    y_profile = construct_profile(y_value)
    # 修正计算误差，最后一项应该是0
    x_profile[-1] = 0
    y_profile[-1] = 0

    writer = csv.writer(open("data/new time series.csv", "w", newline=""))
    writer.writerow(["X(t)", "Y(t)"])
    for i, j in zip(x_profile, y_profile):
        writer.writerow([str(i), str(j)])

    temp_array = np.arange(2.75, 5.6, 0.25)
    s_array = [int(np.exp(i)) for i in temp_array]
    Fqs_result = []
    h_xyq_results = []
    h_xyq_derivative_result = []
    q_array = np.array([i for i in range(-q, q + 1, 2)])
    interval = 0.1
    q_another_array = np.array([i + interval for i in q_array])  # 用来求斜率
    for s in s_array:
        # 得到分割后的（2Ns * s）序列
        x_division = divide_profile(x_profile, s)
        y_division = divide_profile(y_profile, s)

        # 得到多项式拟合函数
        x_polynomial_func_array, x_rmse = fit_polynomial(x_division, m)
        y_polynomial_func_array, y_rmse = fit_polynomial(y_division, m)

        # 得到detrended covariance F2(s,v)
        F2 = compute_covariance(x_polynomial_func_array, x_division, y_polynomial_func_array, y_division)

        # 记录每一个多项式拟合的参数和误差， 需要的话可以取消注释
        # writer = csv.writer(open("data/s=" + str(s) + ".csv", "w", newline=""))
        # writer.writerow(["x_segment", "x_polynomial", "x_rmse", "y_segment", "y_polynomial", "y_rmse", "F2(s,v)"])
        # for i in range(len(x_division)):
        #     writer.writerow([str(x_division[i]), str(x_polynomial_func_array[i].coefficients), str(x_rmse[i]),
        #                      str(y_division[i]), str(y_polynomial_func_array[i].coefficients), str(y_rmse[i]),
        #                      str(F2[i])])

        Fqs = []
        for i in q_array:
            Fqs.append(fluctuation_func(F2, i))

        # q + interval
        Fqs_another = []
        for i in q_another_array:
            Fqs_another.append(fluctuation_func(F2, i))

        Fqs_result.append(Fqs)
        # 计算h_xy(q), 忽略常数logC
        h_xyq = np.log(Fqs)/np.log(s)
        h_xyq_results.append(h_xyq)

        h_xyq_another = np.log(Fqs_another)/np.log(s)
        # h_xy(q)在每一个q值的导数（斜率）
        h_xyq_derivative_result.append((h_xyq_another-h_xyq)/interval)

    # 记录h_xy(q)
    writer = csv.writer(open("data/h_xy(q).csv", "w", newline=""))
    q_head_array = ["q=" + str(i) for i in range(-q, q + 1, 2)]
    head = [""]
    head.extend(q_head_array)
    writer.writerow(head)
    for i in range(len(s_array)):
        row = ["s=" + str(s_array[i])]
        temp = [str(j) for j in h_xyq_results[i]]
        row.extend(temp)
        writer.writerow(row)

    h_xyq_results = np.array(h_xyq_results)
    h_xyq_derivative_result = np.array(h_xyq_derivative_result)
    # 计算tau_xy(q)
    tau_xyq_results = q_array * h_xyq_results - 1
    # 计算alpha_xy(q)
    alpha_xy_result = h_xyq_results + q_array * h_xyq_derivative_result
    # 计算f_xy(alpha)
    f_xy_alpha = q_array * (alpha_xy_result - h_xyq_results) + 1

    # 记录alpha_xy的值
    writer = csv.writer(open("data/alpha_xy(q).csv", "w", newline=""))
    q_head_array = ["q=" + str(i) for i in range(-q, q + 1, 2)]
    head = [""]
    head.extend(q_head_array)
    writer.writerow(head)
    for i in range(len(s_array)):
        row = ["s=" + str(s_array[i])]
        temp = [str(j) for j in alpha_xy_result[i]]
        row.extend(temp)
        writer.writerow(row)

    # tao_xy(q)关于q的图
    pl.figure(figsize=(10, 7))
    for index, row in enumerate(tau_xyq_results):
        pl.plot(q_array, row, label=s_array[index])
    pl.xlabel("q")
    pl.ylabel("tau_xy(q)")
    pl.title("m=" + str(m))
    pl.legend()
    # pl.show()
    pl.savefig("data/tau_xy(q).png")

    # f_xy(alpha)关于alpha的图
    pl.figure(figsize=(10, 7))
    for index, row in enumerate(f_xy_alpha):
        pl.plot(alpha_xy_result[index], row, label=s_array[index])
    pl.xlabel("alpha")
    pl.ylabel("f_xy(alpha)")
    pl.title("m=" + str(m))
    pl.legend()
    # pl.show()
    pl.savefig("data/f_xy(alpha).png")

    # h_xy(q)关于q的图
    s_array = ["s=" + str(i) for i in s_array]
    pl.figure(figsize=(10, 7))
    for index, row in enumerate(h_xyq_results):
        pl.plot(q_array, row, label=s_array[index])
    pl.xlabel("q")
    pl.ylabel("h_xy(q)")
    pl.title("m=" + str(m))
    pl.legend()
    # pl.show()
    pl.savefig("data/h_xy(q).png")

