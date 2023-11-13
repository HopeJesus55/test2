import random
import math


class SV:
    """класс - случаные величины
        @ count - мощность выборки
        @ numbers - список для "базовых" случаных чисел"""

    def __init__(self, count):
        """Метод инициализации. Свойства:
            @ count - мощность выборки"""
        self.count = count
        self.numbers = []

    def _generate_random_base_numbers(self):
        """Метод генератор "базовых" случайных величин.
            Генерирует count величин и записывает в список numbers
            @ Возвращает список numbers"""
        i = 0
        while i < self.count:
            self.numbers.append(random.random())
            i+=1
        return self.numbers


class ExpSV(SV):
    """класс-наследник - экспоненциальное распределение
        @ count - мощность выборки
        @ Lambda - лямбда, 1/среднее выборки (1/Tsp)
        @ rasp - список полученных экспоненциально-распределенных величин"""

    def __init__(self, count, Lambda):
        """Метод инициализации. Свойства:
            @ count - мощность выборки
            @ Lambda - лямбда, 1/среднее выборки (1/Tsp)"""
        super().__init__(count)
        self.Lambda = Lambda
        self.rasp = []

    def generate_exp_rasp(self):
        """Метод генератор экспоненциально-распределенных величин.
        @ Возвращает rasp - список экспоненц.-распр. величин"""
        self.numbers = self._generate_random_base_numbers()
        for n in self.numbers:
            self.rasp.append(-(math.log(n) / float(self.Lambda)))
        return self.rasp


class NormSV(SV):
    """класс-наследник - нормальное распределение
        @ count - мощность выборки
        @ Tsp - среднее выборки
        @ Sigma - среднеквадратичное отклонение
        @ rasp - список полученных нормально-распределенных величин"""

    def __init__(self, count, Tsp, Sigma):
        """Метод инициализации. Свойства:
            @ count - мощность выборки
            @ Tsp - среднее выборки
            @ Sigma - среднеквадратичное отклонение"""
        super().__init__(count)
        self.Tsp = Tsp
        self.Sigma = Sigma
        self.rasp = []

    def generate_norm_rasp(self):
        """Метод генератор нормально-распределенных величин.
            @ Возвращает rasp - список нормально-распр. величин"""
        r1 = self._generate_random_base_numbers()
        for r in r1:
            r2 = random.random()
            temp = math.cos(2 * math.pi * r) * math.sqrt(-2 * math.log(r2, math.e))
            self.rasp.append(self.Tsp + self.Sigma * temp)
        return self.rasp
