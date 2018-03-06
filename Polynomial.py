

class Polynomial:
    def __init__(self, parameters):
        self.degree = len(parameters)-1  # 最高次项
        self.parameters = parameters  # 各次项由低到高的系数
